import math
import os
import wandb
import numpy as np
import torch
from einops import rearrange
from geqtrain.data import AtomicData, AtomicDataDict
from source.utils.mol_utils import coords_atomicnum_to_mol, align_mols
from tqdm import tqdm
from source.diffusion_utils.utils import (
    sample_noise_from_N_0_1,
    cdf_standard_gaussian,
    sigma,
    alpha,
    center_pos,
)

# reference for diffusion-free guidance: https://github.com/ncchaudhari10/Classifier-Free-Diffusion/blob/main/inference.py

def fetch(trainer):

    def fetch_diffusion_module(model, diffusion_module_name:str='diffusion_module'):
        for k, m in model.named_children():
            if k == diffusion_module_name:
                return m
        return None

    model = trainer.model
    diff_module = fetch_diffusion_module(model)
    if not diff_module: raise ValueError(f"Diffusion module not found in model: {model}")

    device = trainer.device
    TMax = diff_module.T
    log_dir = trainer.output.workdir

    atom_types = trainer.dataset_val[0]['node_types'].squeeze().to(device)

    labels_conditioned = diff_module.labels_conditioned
    number_of_labels = diff_module.number_of_labels

    return model, diff_module, device, TMax, atom_types, log_dir, labels_conditioned, number_of_labels

def get_noise_pred(model, t_tensor, x_t, atom_types, condition_class:int, labels_conditioned:bool, number_of_labels:int, guidance_scale:float):
    #! EDGE_VECTORS_KEY, EDGE_LENGTH_KEY are always recomputed inside the fwd of model
    bs = t_tensor.shape[0]
    args = {
        't_inference':                t_tensor, # passed to know what t to embed for conditioning
        AtomicDataDict.POSITIONS_KEY: x_t,
        AtomicDataDict.R_MAX_KEY:     torch.finfo(torch.float32).max,
        AtomicDataDict.BATCH_KEY:     torch.zeros((x_t.shape[0], 1), device=t_tensor.device),
        AtomicDataDict.NODE_TYPE_KEY: atom_types,

    }
    args['labels'] = torch.full((bs, 1), number_of_labels-1,  dtype=torch.int64, device=t_tensor.device) # last label for no conditioning if not labels_conditioned then not used in fwd
    # args['labels'] = torch.tensor(number_of_labels-1,  dtype=torch.int64, device=t_tensor.device) # last label for no conditioning if not labels_conditioned then not used in fwd
    data = AtomicData.from_points(**args)
    out = model(AtomicData.to_AtomicDataDict(data))
    uncoditioned_eps_pred = out['noise']

    if not labels_conditioned:
        return uncoditioned_eps_pred

    # conditioned eps prediction
    args['labels'] = torch.full((bs, 1), condition_class,  dtype=torch.int64, device=t_tensor.device) # last label for no conditioning if not labels_conditioned then not used in fwd
    data = AtomicData.from_points(**args)
    out = model(AtomicData.to_AtomicDataDict(data))
    coditioned_eps_pred = out['noise']

    # guided_noise = uncoditioned_eps_pred + guidance_scale * (coditioned_eps_pred - uncoditioned_eps_pred)
    # (1 + w) * eps_cond - w * eps_uncond
    w = guidance_scale
    guided_noise = (1 + w) * coditioned_eps_pred - w * uncoditioned_eps_pred

    return guided_noise

class DiffusionSamplerLogger(object):
    def __init__(self, log_dir:str, sampler_type:str):
        self.log_dir = os.path.join(log_dir, sampler_type)
        os.makedirs(self.log_dir, exist_ok=True)

    def log(self, epoch, sample_idx, positions_for_gif, condition_class, gscale, tmax, atom_types):
        sampling_log = os.path.join(self.log_dir, f"pos_per_tmax_{tmax}_epoch_{epoch}_sampleID_{sample_idx}_label_{condition_class}_gscale_{gscale}.log")
        with open(sampling_log, 'w') as f:
            f.write(f"Atom types: {atom_types.tolist()}\n")
            for t_idx, pos in enumerate(positions_for_gif):
                f.write(f"Timestep: {t_idx}\n")
                for atom_idx, atom_pos in enumerate(pos):
                    f.write(f"{atom_idx}: {atom_pos.tolist()}\n")
                f.write("\n")


@torch.no_grad()
def ddim_sampling(
    trainer,
    method="quadratic",
    n_steps:int = 50,
    t_init:int = 0,
    condition_class:int = 0,
    guidance_scale:float = 7.0,
    forced_log_dir:str=None,
    iter_epochs:int=5,
    n_samples:int=1,
): # n_steps: how many t to do in sampling

    if trainer.iepoch % iter_epochs != 0: return
    model, diff_module, device, TMax, atom_types, log_dir, labels_conditioned, number_of_labels = fetch(trainer)
    if t_init >= TMax: raise ValueError(f"t_init ({t_init}) must be less than TMax ({TMax})")
    logger = DiffusionSamplerLogger(forced_log_dir if forced_log_dir else log_dir, 'ddim_sampler')

    alpha_bar = diff_module.noise_scheduler.alpha_bar

    def get_time_steps(method, n_steps):
        if method == "linear":
            step_size = TMax // n_steps
            time_steps = np.asarray(list(range(0, TMax, step_size)), dtype=int)
        elif method == "quadratic":
            time_steps = (np.linspace(0, np.sqrt(TMax * 0.8), n_steps) ** 2).astype(int)
        else:
            raise NotImplementedError(f"sampling method {method} is not implemented!")
        return time_steps

    time_steps = get_time_steps(method, n_steps)
    time_steps = time_steps + 1
    time_steps_prev = np.concatenate([[0], time_steps[:-1]])

    for sample_idx in tqdm(range(n_samples), desc="Sampling"):
        # atom_types = atom_types[torch.randperm(atom_types.size(0))]

        # initial rand coords sampled from prior
        x_t = center_pos((sample_noise_from_N_0_1(size=(atom_types.shape[0], 3), device=device)))
        initial_rand_pos = x_t.clone().detach()
        positions_for_gif = [initial_rand_pos]

        for i in tqdm(reversed(range(t_init, n_steps)), desc=f"DDIM Sampling Progress for sample {sample_idx}"):
            # get t and prev_t
            t = time_steps[i]
            t_tensor = torch.full((1,), t, dtype=torch.long, device=device)
            t_prev = time_steps_prev[i]
            # t_prev_tensor = torch.ones(n_samples, dtype=torch.long, device=device) * t_prev

            # get alphabar and alphabar prev
            alpha_bar_t = alpha_bar[t]
            alpha_t_prev = alpha_bar[t_prev]

            eps_pred = get_noise_pred(
                model,
                t_tensor,
                x_t,
                atom_types,
                condition_class=condition_class,
                labels_conditioned=labels_conditioned,
                number_of_labels=number_of_labels,
                guidance_scale=guidance_scale,
            )

            # DDIM update rule
            eta = 0.0
            sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_t_prev))

            # sample noise if not t=0
            if i > 1:
                epsilon_t = center_pos((sample_noise_from_N_0_1(size=(atom_types.shape[0], 3), device=device)))
            else:
                epsilon_t = torch.zeros(size=(atom_types.shape[0], 3), device=device)

            x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
            dir_xt = torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) * eps_pred
            x_t = torch.sqrt(alpha_t_prev) * x0_pred + dir_xt + sigma_t * epsilon_t
            x_t = center_pos(x_t)
            positions_for_gif.append(x_t.clone().detach())

        logger.log(trainer.iepoch, sample_idx, positions_for_gif, condition_class, guidance_scale, n_steps, atom_types)


@torch.no_grad()
def ddpm_sampling(
    trainer,
    t_init:int = 0,
    condition_class:int = 0,
    guidance_scale:float = 7.0,
    n_samples:int=1,
    iter_epochs:int=5,
):
    if trainer.iepoch % iter_epochs != 0: return
    model, diff_module, device, TMax, atom_types, log_dir, labels_conditioned, number_of_labels = fetch(trainer)
    if t_init >= TMax: raise ValueError(f"t_init ({t_init}) must be less than TMax ({TMax})")
    logger = DiffusionSamplerLogger(log_dir, 'ddpm_sampler')

    # fetch params for quick access from diff model
    one_minus_alphas = diff_module.noise_scheduler.one_minus_alphas
    sqrt_alphas = diff_module.noise_scheduler.sqrt_alphas
    sqrt_one_minus_alpha_bar = diff_module.noise_scheduler.sqrt_one_minus_alpha_bar
    sqrt_betas = diff_module.noise_scheduler.sqrt_betas

    for sample_idx in range(n_samples):
        # atom_types = atom_types[torch.randperm(atom_types.size(0))]

        # initial rand coords sampled from prior
        x_t = center_pos((sample_noise_from_N_0_1(size=(atom_types.shape[0], 3), device=device)))
        initial_rand_pos = x_t.clone().detach()
        positions_for_gif = [initial_rand_pos]

        for t in tqdm(reversed(range(t_init, TMax)), desc="DDPM Sampling Progress"):

            # get noise in x_t
            epsilon_pred = get_noise_pred(
                model,
                torch.tensor([t], device=device),
                x_t,
                atom_types,
                condition_class=condition_class,
                labels_conditioned=labels_conditioned,
                number_of_labels=number_of_labels,
                guidance_scale=guidance_scale,
            )

            # get coefficients for x_t-1 calculation
            _sqrt_alpha_t = sqrt_alphas[t].to(device)
            _one_minus_alpha_t = one_minus_alphas[t].to(device)
            _sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar[t].to(device)
            _sqrt_betas = sqrt_betas[t].to(device)

            # sample noise if not t=0
            if t > 1:
                z = center_pos((sample_noise_from_N_0_1(size=(x_t.shape[0], 3), device=device)))
            else:
                z = torch.zeros(size=(x_t.shape[0], 3), device=device)

            x_t = 1 / _sqrt_alpha_t * (x_t - (_one_minus_alpha_t/_sqrt_one_minus_alpha_bar_t)*epsilon_pred) + _sqrt_betas*z
            x_t = center_pos(x_t)
            positions_for_gif.append(x_t.clone().detach())

        logger.log(trainer.iepoch, sample_idx, positions_for_gif, condition_class, guidance_scale, TMax, atom_types)





















##############################################################################################################################
##############################################################################################################################

# @torch.no_grad()
# def generate_single_mol( #! let's start from generating 1 mol at the time
#     trainer,
#     n_nodes=15,
#     n_features=22,
#     context=None,
#     n_samples=1,
#     fix_noise=False,
# ):

#     if trainer.iepoch % 5 != 0:
#         return

#     model  = trainer.model
#     device = trainer.device
#     diffusion_module = fetch_diffusion_module(model)

#     for mol_idx in range(2):

#         z_x = center_pos((sample_noise_from_N_0_1(size=(n_nodes, 3), device=device)))
#         z_h = sample_noise_from_N_0_1(size=(n_nodes, n_features), device=device)

#         # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
#         x_means_diff = []
#         h_means_diff = []
#         for s in reversed(range(0, diffusion_module.T)):
#             s_array = torch.full((n_samples, 1), fill_value=s, device=device) # it can be that this must be a float in this case
#             t_array = s_array + 1
#             s_array = s_array / diffusion_module.T
#             t_array = t_array / diffusion_module.T
#             z_x, z_h = diffusion_module.sample_p_zs_given_zt(
#                 s_array,
#                 t_array,
#                 z_x,
#                 z_h,
#                 model,
#                 context,
#                 fix_noise=fix_noise
#             )
#             x_means_diff.append(z_x.mean().item())
#             h_means_diff.append(z_h.mean().item())
#         # Finally sample p(x, h | z_0).
#         x, h = diffusion_module.sample_p_xh_given_z0(z_x, z_h, model, context, fix_noise=fix_noise)

#         # diffusion_utils.assert_mean_zero_with_mask(x, node_mask)

#         max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
#         if max_cog > 5e-2:
#             print(f'Warning cog drift with error {max_cog:.3f}. Projecting the positions down.')
#             x = center_pos(x) # diffusion_utils.remove_mean_with_mask(x, node_mask)

#         # Create log directory if it doesn't exist
#         log_dir = "/storage_common/nobilm/diffusion/log"
#         os.makedirs(log_dir, exist_ok=True)

#         # Create log file path with epoch and molecule index
#         log_file = os.path.join(log_dir, f"generated_mol_epoch_{trainer.iepoch}_mol_{mol_idx}.log")

#         # Calculate statistics
#         x_mean = x.mean().item()
#         x_std = x.std().item()

#         # Write to log file
#         with open(log_file, f'w') as f:
#             f.write(f"x: {x.tolist()}\n")
#             f.write(f"h: {h.tolist()}\n")
#             f.write(f"x mean: {x_mean:.6f}, x std: {x_std:.6f}\n")
#             f.write(f"x_means_diff: {[m for m in x_means_diff]}\n")
#             f.write(f"h_means_diff: {[m for m in h_means_diff]}\n")
#             f.write(f"max COG: {max_cog}\n")





'''if True: # fix_noise:
    # Noise is broadcasted over the batch axis, useful for visualizations.
    z = self.sample_combined_position_feature_noise((n_nodes, 3), (n_nodes, n_features), device) # z_T intially
else:
    z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask) # z_T intially    '''



# def get_original_mol_info():
#     # atom_types =torch.tensor([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 7], dtype=torch.int64) # single_mol_source
#     # og_pos = torch.tensor(
#     #     [[ 3.3185,  0.7458, -1.7019],
#     #      [ 9.898 ,  1.3086,  1.5853],
#     #      [ 5.6092, -1.5001,  2.1069],
#     #      [ 9.0685, -4.0192,  1.0114],
#     #      [ 4.4373,  1.7569, -1.9705],
#     #      [ 5.2042,  2.1847, -0.7146],
#     #      [ 6.6583,  1.4447,  1.1073],
#     #      [ 8.1482, -3.0112,  2.9472],
#     #      [ 8.873 ,  0.2469,  1.856 ],
#     #      [ 6.9953, -0.937 ,  2.0719],
#     #      [ 7.4638,  0.3007,  1.6554],
#     #      [ 5.8896,  1.0576, -0.0834],
#     #      [ 9.2485, -0.9266,  2.359 ],
#     #      [ 8.0918, -1.6349,  2.4969],
#     #      [ 7.9767, -3.9565,  1.9213]], dtype=torch.float32)

#     atom_types = torch.tensor([6, 6, 8, 7, 6, 6, 6, 8, 7, 6], dtype=torch.int64)
#     og_pos = torch.tensor(
#         [[ 3.72      , 26.57      ,  2.11      ],
#         [ 2.77      , 25.8       ,  1.23      ],
#         [ 1.5999999 , 26.15      ,  1.0899999 ],
#         [ 3.27      , 24.64      ,  0.69      ],
#         [ 2.48      , 23.689999  , -0.19      ],
#         [ 3.47      , 23.16      , -1.27      ],
#         [ 1.7299999 , 22.59      ,  0.48999998],
#         [ 2.34      , 21.88      ,  1.2800001 ],
#         [ 0.39999998, 22.43      ,  0.21      ],
#         [-0.46999997, 21.35      ,  0.73      ]], dtype=torch.float32)

#     og_mol = coords_atomicnum_to_mol(coords=og_pos, atomic_num=atom_types, removeHs=True)
#     return atom_types, og_pos, og_mol