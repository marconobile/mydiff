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

@torch.no_grad()
def generate_single_mol( #! let's start from generating 1 mol at the time
    trainer,
    n_nodes=10,
    n_features=22,
    context=None,
    n_samples=1,
    fix_noise=False,
):
    model  = trainer.model
    device = trainer.device
    diffusion_module = fetch_diffusion_module(model)

    z_x = center_pos((sample_noise_from_N_0_1(size=(n_nodes, 3), device=device)))
    z_h = sample_noise_from_N_0_1(size=(n_nodes, n_features), device=device)

    # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
    for s in reversed(range(0, diffusion_module.T)):
        s_array = torch.full((n_samples, 1), fill_value=s, device=device) # it can be that this must be a float in this case
        t_array = s_array + 1
        s_array = s_array / diffusion_module.T
        t_array = t_array / diffusion_module.T
        z_x, z_h = diffusion_module.sample_p_zs_given_zt(
            s_array,
            t_array,
            z_x,
            z_h,
            model,
            context,
            fix_noise=fix_noise
        )

    # Finally sample p(x, h | z_0).
    x, h = diffusion_module.sample_p_xh_given_z0(z_x, z_h, model, context, fix_noise=fix_noise)

    # diffusion_utils.assert_mean_zero_with_mask(x, node_mask)

    max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
    if max_cog > 5e-2:
        print(f'Warning cog drift with error {max_cog:.3f}. Projecting the positions down.')
        x = center_pos(x) # diffusion_utils.remove_mean_with_mask(x, node_mask)

    return x, h





'''if True: # fix_noise:
    # Noise is broadcasted over the batch axis, useful for visualizations.
    z = self.sample_combined_position_feature_noise((n_nodes, 3), (n_nodes, n_features), device) # z_T intially
else:
    z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask) # z_T intially    '''