# reference for diffusion-free guidance: https://github.com/ncchaudhari10/Classifier-Free-Diffusion/blob/main/inference.py

import os
import numpy as np
import torch
from geqtrain.data import AtomicData, AtomicDataDict
from source.utils.mol_utils import coords_atomicnum_to_mol
from tqdm import tqdm
from source.diffusion_utils.utils import sample_noise_from_N_0_1, center_pos
from rdkit.Chem import AllChem
import shutil
from pathlib import Path
import matplotlib.pyplot as plt


def guidance_scale_scheduler(steps, min_guidance=0.0, max_guidance=4.0):
    half = torch.linspace(max_guidance, min_guidance, steps)
    # Create a symmetric scheduler that stays at min_guidance for half of the steps
    steps = half.shape[0]
    hold_steps = steps // 2

    # First half: max_guidance → min_guidance
    down = torch.linspace(max_guidance, min_guidance, steps // 4) # 1/4 in a, 1/4 to go to b, 1/2 to hold to min_guidance
    # Middle: hold at min_guidance
    hold = min_guidance * torch.ones(hold_steps)
    # Last half: min_guidance → max_guidance
    up = torch.linspace(min_guidance, max_guidance, steps // 4)

    # Concatenate all parts
    return torch.cat([down, hold, up])

def guidance_scale_scheduler_flat(steps, min_guidance=0.0, max_guidance=4.0):
    # First chunk: exponential decay from max_guidance to min_guidance, occupies 1/5 of steps
    n_exp = steps // 7
    exp_decay = max_guidance * torch.exp(torch.linspace(0, -3, n_exp))
    exp_decay = (exp_decay - exp_decay[-1]) / (exp_decay[0] - exp_decay[-1]) * (max_guidance - min_guidance) + min_guidance

    # Middle: hold at min_guidance
    # n_hold = steps // 2
    # hold = min_guidance * torch.ones(n_hold)

    # Last chunk: linear up from min_guidance to max_guidance, occupies remaining steps
    n_up = steps - n_exp # - n_hold
    up = torch.linspace(min_guidance, max_guidance, n_up)
    up = torch.zeros_like(up)

    return torch.cat([exp_decay, up])


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


@torch.amp.autocast('cuda', enabled=False)
@torch.no_grad()
def get_noise_pred(
    model,
    t_tensor,
    x_t,
    atom_types,
    condition_class:int,
    labels_conditioned:bool,
    number_of_labels:int,
    guidance_scale:float,
    target_lbl:int|None=None,
    interpolation_coeff=None,
    alpha = 0.0, # 0 rigid, 0.9 funky, 0.0 as deactivated
):
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

    if target_lbl is not None:
        target_lbl = torch.full((bs, 1), target_lbl,  dtype=torch.int64, device=t_tensor.device)
        args['target_lbl'] = target_lbl
        args['target_interpolation_coeff'] = interpolation_coeff

    data = AtomicData.from_points(**args)
    out = model(AtomicData.to_AtomicDataDict(data))
    coditioned_eps_pred = out['noise']

    w = guidance_scale
    # guided_noise = (1 + w) * coditioned_eps_pred - w * uncoditioned_eps_pred
    guided_noise = coditioned_eps_pred + w * (coditioned_eps_pred - uncoditioned_eps_pred) * torch.abs(coditioned_eps_pred - uncoditioned_eps_pred).pow(alpha)
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


def get_time_steps(method, n_steps, TMax):
    if method == "linear":
        step_size = TMax // n_steps
        time_steps = np.asarray(list(range(0, TMax, step_size)), dtype=int)
    elif method == "quadratic":
        time_steps = (np.linspace(0, np.sqrt(TMax * 0.8), n_steps) ** 2).astype(int)
    else:
        raise NotImplementedError(f"sampling method {method} is not implemented!")
    return time_steps


@torch.amp.autocast('cuda', enabled=False)
@torch.no_grad()
def ddim_sampling(
    trainer,
    method="quadratic",
    n_steps:int = 50,
    t_init:int = 0,
    condition_class:int = 0,
    guidance_scale:float = 7.0,
    forced_log_dir:str=None,
    iter_epochs:int=1,
    n_samples:int=1,
): # n_steps: how many t to do in sampling

    if trainer.iepoch % iter_epochs != 0: return
    model, diff_module, device, TMax, atom_types, log_dir, labels_conditioned, number_of_labels = fetch(trainer)
    if t_init >= TMax: raise ValueError(f"t_init ({t_init}) must be less than TMax ({TMax})")
    logger = DiffusionSamplerLogger(forced_log_dir if forced_log_dir else log_dir, 'ddim_sampler')

    alpha_bar = diff_module.noise_scheduler.alpha_bar
    time_steps = get_time_steps(method, n_steps, TMax)
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
        if n_samples == 1: return x_t


@torch.amp.autocast('cuda', enabled=False)
@torch.no_grad()
def ddpm_sampling(
    trainer,
    t_init:int = 0,
    condition_class:int = 0,
    guidance_scale:float = 7.0,
    n_samples:int=1,
    iter_epochs:int=5,
    forced_log_dir:str=None,
    initial_rand_pos:torch.Tensor=None, # if not None, will use this as initial position instead of sampling from prior
    force_tmax:int=-1, # if not None, will use this as TMax instead of the one from the diffusion module
    target_lbl=None,
    interpolation_coeff=None, # if not None, will use this for conditioning interpolation
):
    if trainer.iepoch % iter_epochs != 0: return
    model, diff_module, device, TMax, atom_types, log_dir, labels_conditioned, number_of_labels = fetch(trainer)
    if t_init >= TMax: raise ValueError(f"t_init ({t_init}) must be less than TMax ({TMax})")
    logger = DiffusionSamplerLogger(forced_log_dir if forced_log_dir else log_dir, 'ddpm_sampler')

    if force_tmax >= 0:
        TMax = force_tmax
        if t_init >= TMax: raise ValueError(f"t_init ({t_init}) must be less than TMax ({TMax})")

    # fetch params for quick access from diff model
    one_minus_alphas = diff_module.noise_scheduler.one_minus_alphas
    sqrt_alphas = diff_module.noise_scheduler.sqrt_alphas
    sqrt_one_minus_alpha_bar = diff_module.noise_scheduler.sqrt_one_minus_alpha_bar
    sqrt_betas = diff_module.noise_scheduler.sqrt_betas

    for sample_idx in range(n_samples):
        # atom_types = atom_types[torch.randperm(atom_types.size(0))]

        if initial_rand_pos is not None:
            if initial_rand_pos.shape[0] != atom_types.shape[0]:
                raise ValueError(f"initial_rand_pos must have the same number of atoms as atom_types ({atom_types.shape[0]}), got {initial_rand_pos.shape[0]}")
            x_t = center_pos(initial_rand_pos.clone().detach())
        else:
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
                target_lbl=target_lbl,
                interpolation_coeff=interpolation_coeff
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
        if n_samples == 1: return x_t


@torch.amp.autocast('cuda', enabled=False)
@torch.no_grad()
def single_DDPM_step(
    atom_types,
    device,
    x_t,
    t,
    model,
    condition_class,
    labels_conditioned,
    number_of_labels,
    guidance_scale,
    diff_module,
    target_lbl=None,
    interpolation_coeff=None, # if not None, will use this for conditioning interpolation
    alpha=0.0,
):
    # fetch params for quick access from diff model
    one_minus_alphas = diff_module.noise_scheduler.one_minus_alphas
    sqrt_alphas = diff_module.noise_scheduler.sqrt_alphas
    sqrt_one_minus_alpha_bar = diff_module.noise_scheduler.sqrt_one_minus_alpha_bar
    sqrt_betas = diff_module.noise_scheduler.sqrt_betas

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
        target_lbl=target_lbl,
        interpolation_coeff=interpolation_coeff,
        alpha=alpha,
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
    return center_pos(x_t)


def rmsd(x_t, ref_mol, atom_types, itr, log_file):
    _rmsd = 42
    if x_t is None: # if x_clean is None, we are still in the first iteration
        return _rmsd
    try:
        mol_t = coords_atomicnum_to_mol(x_t, atom_types, sanitize=False)
        _rmsd = AllChem.GetBestRMS(mol_t, ref_mol)
    except:
        pass
    print(f"itr {itr}, rmsd: {_rmsd}")
    if _rmsd ==  42: exit()
    return _rmsd


def rm_files(fd):
    if os.path.exists(fd):
        for filename in os.listdir(fd):
            file_path = os.path.join(fd, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)


def get_unique_exp_name(base_dir, exp_name):
    """
    Returns a unique experiment name by appending a sequential integer if needed.
    """
    base_exp_dir = Path(base_dir)
    exp_dir = base_exp_dir / exp_name
    if exp_dir.exists():
        i = 1
        while (base_exp_dir / f"{exp_name}_{i}").exists():
            i += 1
        exp_name = f"{exp_name}_{i}"
    return exp_name


@torch.no_grad()
@torch.amp.autocast('cuda', enabled=False)
def sample_alanine_transition_pathDDPM(
    trainer,
    condition_class:int = 0,
    guidance_scale:float = 7.0,
    forced_log_dir:str=None,
):
    # C5: 0
    # PII: 1
    # alphaP: 2 WRONG
    # alphaR: 3
    # C7ax: 4
    # alphaL: 5

    start_state_category = 0
    target_state_category = 4

    model, diff_module, device, TMax, atom_types, log_dir, labels_conditioned, number_of_labels = fetch(trainer)
    exp_name = 'TPS_from_{}_to_{}'.format(start_state_category, target_state_category)

    base_exp_dir = forced_log_dir if forced_log_dir else log_dir
    exp_name = get_unique_exp_name(base_exp_dir, exp_name)
    logger = DiffusionSamplerLogger(base_exp_dir, exp_name)

    # generate start state
    start_pos = ddim_sampling(
        trainer,
        iter_epochs=1,
        condition_class = start_state_category,
        forced_log_dir=str(Path(logger.log_dir) / 'start_state'),
    )

    # generate target state
    target_pos = ddim_sampling(
        trainer,
        iter_epochs=1,
        condition_class = target_state_category,
        forced_log_dir=str(Path(logger.log_dir) / 'target_state'),
    )
    ref_mol = coords_atomicnum_to_mol(target_pos, atom_types, sanitize=False)

    #! hyperparameters
    eval_every = 50
    treshold = .25
    t = 10
    path_len = 50000
    coeffs = torch.linspace(0, 1, path_len)
    min_guidance = 0.0
    guidance_scale = 7.0
    # guidance_scale_itr = guidance_scale_scheduler_flat(path_len,  min_guidance=min_guidance, max_guidance=guidance_scale)
    alpha = 0.9


    plt.figure()
    # plt.plot(guidance_scale_itr.cpu().numpy())
    plt.title("Guidance Scale Scheduler")
    plt.xlabel("Step")
    plt.ylabel("Guidance Scale")
    plt.savefig(str(Path(logger.log_dir) / 'guidance_scheduler.png'))
    plt.close()
    # np.savetxt(str(Path(logger.log_dir) / 'guidance_scheduler.txt'), guidance_scale_itr.cpu().numpy())

    # Plot and save interpolation_coeff pattern
    plt.figure()
    plt.plot(coeffs.cpu().numpy())
    plt.title("Interpolation Coefficient")
    plt.xlabel("Step")
    plt.ylabel("Interpolation Coefficient")
    plt.savefig(str(Path(logger.log_dir) / 'interpolation_coeff.png'))
    plt.close()
    np.savetxt(str(Path(logger.log_dir) / 'interpolation_coeff.txt'), coeffs.cpu().numpy())

    # log params
    params_log_file = str(Path(logger.log_dir) / 'params.txt')
    with open(params_log_file, 'w') as f:
        f.write(f"start_state_category: {start_state_category}\n")
        f.write(f"target_state_category: {target_state_category}\n")
        f.write(f"t: {t}\n")
        f.write(f"treshold: {treshold}\n")
        f.write(f"eval_every: {eval_every}\n")
        f.write(f"path_len: {path_len}\n")
        f.write(f"max guidance_scale: {guidance_scale}\n")
        f.write(f"min guidance_scale: {min_guidance}\n")
        f.write(f"alpha: {alpha}\n")

    x_clean = None
    itr = 0
    x_tmp = start_pos
    log_file = str(Path(logger.log_dir) / 'rmsd_log.txt')
    interpolation_coeff = coeffs[itr]
    # guidance_scale = guidance_scale_itr[itr]
    while True:
        x_tmp_new = single_DDPM_step( # single FF step
            atom_types=atom_types,
            device=device,
            x_t=x_tmp,
            t=t,
            model=model,
            condition_class=start_state_category,
            labels_conditioned=labels_conditioned,
            number_of_labels=number_of_labels,
            guidance_scale=guidance_scale,
            diff_module=diff_module,
            target_lbl=target_state_category,
            interpolation_coeff=interpolation_coeff,
            alpha=alpha,
        )

        x_clean = None
        if itr % eval_every == 0:
            print(f"iter {itr} - interpolation_coeff: {interpolation_coeff}, guidance_scale: {guidance_scale}")
            x_clean = ddpm_sampling(
                trainer,
                condition_class=start_state_category,
                guidance_scale=guidance_scale,
                iter_epochs=1,
                forced_log_dir=str(Path(logger.log_dir) / 'path' / f'TP_itr_{itr}'),
                initial_rand_pos=x_tmp_new, # if not None, will use this as initial position instead of sampling from prior
                force_tmax=t-1,
                target_lbl=target_state_category,
                interpolation_coeff=interpolation_coeff,
            )

        x_tmp = x_tmp_new
        if x_clean != None:
            rmsd_value = rmsd(x_clean, ref_mol, atom_types, itr, log_file)
            with open(log_file, "a") as f:
                f.write(f"from {start_state_category} to {target_state_category}, itr {itr}, rmsd: {rmsd_value:.3f}, interpolation_coeff {interpolation_coeff}, guidance_scale {guidance_scale}\n")

        itr+=1
        try:
            interpolation_coeff = coeffs[itr]
            # guidance_scale = guidance_scale_itr[itr]
        except:
            interpolation_coeff = coeffs[-1]
            # guidance_scale = guidance_scale_itr[-1] #+ (itr* 0.0001)

        if rmsd_value < treshold:
            break




# @torch.no_grad()
# @torch.amp.autocast('cuda', enabled=False)
# def sample_alanine_transition_pathDDPM_SEQUENITAL(
#     trainer,
#     states_list:list[int],
#     forced_log_dir:str=None,
# ):
#     # C5: 0
#     # PII: 1
#     # alphaP: 2 WRONG
#     # alphaR: 3
#     # C7ax: 4
#     # alphaL: 5

#     start_state_category = states_list[0]
#     target_state_category = states_list[-1]
#     model, diff_module, device, TMax, atom_types, log_dir, labels_conditioned, number_of_labels = fetch(trainer)
#     exp_name = 'TPS_from_{}_to_{}'.format(start_state_category, target_state_category)

#     base_exp_dir = forced_log_dir if forced_log_dir else log_dir
#     exp_name = get_unique_exp_name(base_exp_dir, exp_name)
#     logger = DiffusionSamplerLogger(base_exp_dir, exp_name)

#     # generate start state
#     start_pos = ddim_sampling(
#         trainer,
#         iter_epochs=1,
#         condition_class = start_state_category,
#         forced_log_dir=str(Path(logger.log_dir) / 'start_state'),
#     )

#     #! hyperparameters
#     eval_every = 50
#     treshold = .29
#     t = 8
#     path_len = 7000
#     coeffs = torch.linspace(0, 1, path_len)
#     min_guidance = 0.0
#     guidance_scale = 3.0
#     guidance_scale_itr = guidance_scale_scheduler(path_len,  min_guidance=min_guidance, max_guidance=guidance_scale)

#     # log params
#     log_file = str(Path(logger.log_dir) / 'rmsd_log.txt')
#     params_log_file = str(Path(logger.log_dir) / 'params.txt')
#     with open(params_log_file, 'w') as f:
#         f.write(f"start_state_category: {start_state_category}\n")
#         f.write(f"target_state_category: {target_state_category}\n")
#         f.write(f"t: {t}\n")
#         f.write(f"treshold: {treshold}\n")
#         f.write(f"eval_every: {eval_every}\n")
#         f.write(f"path_len: {path_len}\n")
#         f.write(f"max guidance_scale: {guidance_scale}\n")
#         f.write(f"min guidance_scale: {min_guidance}\n")

#     # Create list of consecutive state transitions as tuples
#     state_pairs = list(zip(states_list[:-1], states_list[1:]))
#     for start_state_category, target_state_category in state_pairs:
#         # generate temp target state
#         target_pos = ddim_sampling(
#             trainer,
#             iter_epochs=1,
#             condition_class = target_state_category,
#             forced_log_dir=str(Path(logger.log_dir) / 'target_state'),
#         )
#         ref_mol = coords_atomicnum_to_mol(target_pos, atom_types, sanitize=False)

#         x_clean = None
#         itr = 0
#         x_tmp = start_pos
#         interpolation_coeff = coeffs[itr]
#         guidance_scale = guidance_scale_itr[itr]

#         # for _ in range(coeffs.shape[0]):
#         while True:
#             x_tmp_new = single_DDPM_step( # single FF step
#                 atom_types=atom_types,
#                 device=device,
#                 x_t=x_tmp,
#                 t=t,
#                 model=model,
#                 condition_class=start_state_category,
#                 labels_conditioned=labels_conditioned,
#                 number_of_labels=number_of_labels,
#                 guidance_scale=guidance_scale,
#                 diff_module=diff_module,
#                 target_lbl=target_state_category,
#                 interpolation_coeff=interpolation_coeff,
#             )

#             x_clean = None
#             if itr % eval_every == 0:
#                 print(f"from {start_state_category} to {target_state_category}, iter {itr} - interpolation_coeff: {interpolation_coeff}, guidance_scale: {guidance_scale}")
#                 x_clean = ddpm_sampling(
#                     trainer,
#                     condition_class=start_state_category,
#                     guidance_scale=guidance_scale,
#                     iter_epochs=1,
#                     forced_log_dir=str(Path(logger.log_dir) / f'TP_itr_{itr}'),
#                     initial_rand_pos=x_tmp_new, # if not None, will use this as initial position instead of sampling from prior
#                     force_tmax=t-1,
#                     target_lbl=target_state_category,
#                     interpolation_coeff=interpolation_coeff,
#                 )

#             x_tmp = x_tmp_new
#             if x_clean != None:
#                 rmsd_value = rmsd(x_clean, ref_mol, atom_types, itr, log_file)
#                 with open(log_file, "a") as f:
#                     f.write(f"from {start_state_category} to {target_state_category}, itr {itr}, rmsd: {rmsd_value:.3f}, interpolation_coeff {interpolation_coeff}, guidance_scale {guidance_scale}\n")

#             itr+=1
#             try:
#                 interpolation_coeff = coeffs[itr]
#                 guidance_scale = guidance_scale_itr[itr]
#             except:
#                 interpolation_coeff = coeffs[-1]
#                 guidance_scale = guidance_scale_itr[-1] #+ (itr* 0.0001)

#             if rmsd_value < treshold:
#                 start_pos = x_tmp
#                 break

###############
# pseudo:
# def sample_alanine_transition_path():
#     x_a = ddim(wn, t = tmax, a)
#     x_b = ddim(wn, t = tmax, b)

#     x_tmp = x_a
#     t = 8 # or 2

#     while( rmsd(x_tmp, x_b) > .3):
#         x_tmp_new = ddim(x_tmp, t, b)
#         norm = l2norm (x_tmp_new, x_tmp)
#         # score norm?
#         x_tmp = x_tmp_new
#         cast xtmp to same atom ordering of xb for correct rmsd calc
#         care i might need to denoise fully sampled struct b4 computing rmsd



# @torch.no_grad()
# def single_DDIM_step(
#     i,
#     atom_types,
#     device,
#     x_t,
#     model,
#     condition_class,
#     labels_conditioned,
#     number_of_labels,
#     guidance_scale,
#     diff_module,
#     time_steps,
#     time_steps_prev,
#     alpha_bar,
# ):
#     # get t and prev_t
#     t = time_steps[i]
#     t_tensor = torch.full((1,), t, dtype=torch.long, device=device)
#     t_prev = time_steps_prev[i]
#     # t_prev_tensor = torch.ones(n_samples, dtype=torch.long, device=device) * t_prev

#     # get alphabar and alphabar prev
#     alpha_bar_t = alpha_bar[t]
#     alpha_t_prev = alpha_bar[t_prev]

#     eps_pred = get_noise_pred(
#         model,
#         t_tensor,
#         x_t,
#         atom_types,
#         condition_class=condition_class,
#         labels_conditioned=labels_conditioned,
#         number_of_labels=number_of_labels,
#         guidance_scale=guidance_scale,
#     )

#     # DDIM update rule
#     eta = 0.0
#     sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_t_prev))

#     # sample noise if not t=0
#     if i > 1:
#         epsilon_t = center_pos((sample_noise_from_N_0_1(size=(atom_types.shape[0], 3), device=device)))
#     else:
#         epsilon_t = torch.zeros(size=(atom_types.shape[0], 3), device=device)

#     x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
#     dir_xt = torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) * eps_pred
#     x_t = torch.sqrt(alpha_t_prev) * x0_pred + dir_xt + sigma_t * epsilon_t
#     return center_pos(x_t)