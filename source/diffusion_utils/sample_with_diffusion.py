import os
import torch
from source.diffusion_utils.utils import (
    sample_noise_from_N_0_1,
    cdf_standard_gaussian,
    sigma,
    alpha,
    center_pos,
)
from geqtrain.data import AtomicData, AtomicDataDict

def fetch_diffusion_module(model, diffusion_module_name:str='diffusion_module'):
    for k, m in model.named_children():
        if k == diffusion_module_name:
            return m

def denoise_at_t_DDPM(t, x_t, atom_types, model):
    device = x_t.device
    if t > 1:
        z = center_pos((sample_noise_from_N_0_1(size=(x_t.shape[0], 3), device=device)))
    else:
        z = center_pos((sample_noise_from_N_0_1(size=(x_t.shape[0], 3), device=device)))
        z = torch.zeros_like(z).to(device)

    # At inference, we use predicted noise(epsilon) to restore perturbed data sample.
    # Get Neural net prediction.
    #! EDGE_VECTORS_KEY, EDGE_LENGTH_KEY are always recomputed inside the fwd of model
    args = {
        't_inference':                t, # passed to know what t to embed for conditioning
        AtomicDataDict.POSITIONS_KEY: x_t,
        AtomicDataDict.R_MAX_KEY:     torch.finfo(torch.float32).max,
        AtomicDataDict.BATCH_KEY:     torch.zeros((x_t.shape[0], 1), device=device),
        AtomicDataDict.NODE_TYPE_KEY: atom_types,

    }
    data = AtomicData.from_points(**args)
    epsilon_pred = model(AtomicData.to_AtomicDataDict(data))

    diff_module = fetch_diffusion_module(model)

    sqrt_alpha = diff_module.gamma.sqrt_alphas.to(device)
    sqrt_alpha = sqrt_alpha[t]

    sqrt_one_minus_alpha_bar = diff_module.gamma.sqrt_one_minus_alpha_bars.to(device)
    sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar[t]

    sqrt_betas = diff_module.gamma.sqrt_betas.to(device)
    sqrt_betas = sqrt_betas[t]

    alphas = diff_module.gamma.alphas.to(device)
    alphas = alphas[t]

    new_pos = 1 / sqrt_alpha * (x_t - (1-alphas)/sqrt_one_minus_alpha_bar*epsilon_pred['noise']) + sqrt_betas*z
    return center_pos(new_pos)


@torch.no_grad()
def generate_single_mol( #! let's start from generating 1 mol at the time
    trainer,
):
    # if trainer.iepoch % 20 != 0:
    #     return

    model  = trainer.model
    device = trainer.device
    diffusion_module = fetch_diffusion_module(model)

    atom_types = torch.tensor([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 7], dtype=torch.int64, device=device)
    og_pos = torch.tensor(
        [[ 3.3185,  0.7458, -1.7019],
         [ 9.898 ,  1.3086,  1.5853],
         [ 5.6092, -1.5001,  2.1069],
         [ 9.0685, -4.0192,  1.0114],
         [ 4.4373,  1.7569, -1.9705],
         [ 5.2042,  2.1847, -0.7146],
         [ 6.6583,  1.4447,  1.1073],
         [ 8.1482, -3.0112,  2.9472],
         [ 8.873 ,  0.2469,  1.856 ],
         [ 6.9953, -0.937 ,  2.0719],
         [ 7.4638,  0.3007,  1.6554],
         [ 5.8896,  1.0576, -0.0834],
         [ 9.2485, -0.9266,  2.359 ],
         [ 8.0918, -1.6349,  2.4969],
         [ 7.9767, -3.9565,  1.9213]], dtype=torch.float32, device=device)

    # initial coords sampled from prior
    z_x = center_pos((sample_noise_from_N_0_1(size=og_pos.shape, device=device)))
    initial_pos_for_log = z_x.clone().detach()

    # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
    for t in reversed(range(0, diffusion_module.T)):
        timestep = torch.tensor([t]).to(device) #.repeat_interleave(1, dim=0).long().to(device) for batched version?
        z_x = denoise_at_t_DDPM(t=timestep, x_t=z_x, atom_types=atom_types, model=model)

    x = z_x
    max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
    if max_cog > 5e-2:
        print(f'Warning cog drift with error {max_cog:.3f}. Projecting the positions down.')
        x = center_pos(x) # diffusion_utils.remove_mean_with_mask(x, node_mask)

    # Create log directory if it doesn't exist
    log_dir = "/storage_common/nobilm/diffusion/Tmax20" # log_pos_ddpm_50_recover"
    os.makedirs(log_dir, exist_ok=True)

    # Create log file path with epoch and molecule index
    log_file = os.path.join(log_dir, f"generated_mol_epoch_{trainer.iepoch}.log")

    # Calculate statistics
    x_mean = x.mean().item()
    x_std = x.std().item()

    # Write to log file
    with open(log_file, f'w') as f:
        f.write(f"OG rand pos: {initial_pos_for_log.tolist()}\n")
        f.write(f"x: {x.tolist()}\n")
        f.write(f"x mean: {x_mean:.6f}, x std: {x_std:.6f}\n")
        f.write(f"max COG: {max_cog}\n")
        pos_diff =torch.norm(og_pos - x, dim=1)
        pos_diff_mean = pos_diff.mean().item()
        f.write(f"Position differences (L2 norm): {pos_diff.tolist()}\n")
        f.write(f"Mean position difference: {pos_diff_mean:.6f}\n")
        f.write(f"Max position difference: {pos_diff.max().item():.6f}\n")



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