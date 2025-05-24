# Taken from https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/equivariant_diffusion/en_diffusion.py
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Union, List
from e3nn import o3
from torch.nn import functional as F
from geqtrain.data import AtomicData, AtomicDataDict
from einops import rearrange
import math
from source.diffusion_utils import (
    sample_noise_from_N_0_1,
    # cdf_standard_gaussian,
    # sigma,
    # alpha,
    center_pos,
    NoiseScheduler,
)
from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin
from source.nn import get_t_embedder


class ForwardDiffusionModule(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        irreps_in:dict={},
        out_field: Optional[str] = 'noise',
        Tmax:int=1000,
        noise_scheduler:str = 'DDPM', # DDPM, polynomial_schedule, cosine_beta_schedule
        t_embedder:str = 'positional', # positional or fourier
        t_embedding_dim:int=256,
    ):
        super().__init__()
        self.out_field = out_field
        self.ref_data_keys = ['noise_target'] # used to register field for ref_data in geqtrain.trainer.batch_step
        self.T = Tmax
        self.t_embedder = get_t_embedder(t_embedder, t_embedding_dim//2, self.T)
        self.noise_scheduler = NoiseScheduler(scheduler=noise_scheduler, T=self.T) # aka gamma in VDM
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={'t_embedded_per_node':o3.Irreps(f'{self.t_embedder.out_size}x0e')}
        )

    @torch.no_grad()
    def _inference_forward(self, data:AtomicDataDict):
        # skip the noisification: it happens in the sampling algo since it is different than the fwd diff step
        t_emb = self.t_embedder(data['t_inference'].unsqueeze(0), data['batch'])
        data['t_embedded_per_node'] = t_emb.squeeze()
        assert data['t_embedded_per_node'].shape == (data[AtomicDataDict.POSITIONS_KEY].shape[0], self.t_embedder.out_size)
        return data

    def forward(self, data:AtomicDataDict):
        if 't_inference' in data:
            return self._inference_forward(data)
        return self._forward(data)

    def _forward(self, data:AtomicDataDict):

        batch_idxs = data['batch']
        device = batch_idxs.device
        bs = len(torch.unique(batch_idxs))

        # sample noise
        x = data[AtomicDataDict.POSITIONS_KEY]
        eps = center_pos((sample_noise_from_N_0_1(size=x.shape, device=device)))

        # sample t, get alpha(t) and sigma(t)
        t = torch.randint(0, self.T, size=(bs, 1), device=device)
        alpha, sigma = self.noise_scheduler(t, batch_idxs, device)

        # compute noised coords and set target, both must be float32
        data[AtomicDataDict.POSITIONS_KEY] = (alpha * x  + sigma * eps).float()
        data['noise_target'] = eps

        # add t-embedding to data obj
        data['t_embedded_per_node'] = self.t_embedder(t, batch_idxs)
        assert data['t_embedded_per_node'].shape == (x.shape[0], self.t_embedder.out_size)
        return data












# data[AtomicDataDict.NODE_ATTRS_KEY] = torch.cat(
#     [data[AtomicDataDict.NODE_ATTRS_KEY],
#     self.t_embedder(t, data['batch'])],
#     dim=-1
# )











############################################################################################


    # def OGforward(self, data:AtomicDataDict):

    #     if 't_inference' in data:
    #         # can't use the normal forward since scheduler params are different for sampling transitions
    #         return self.forward_sampling(data)

    #     x = data['pos'] # option: drop centering transform and center data['pos'] here considering batch
    #     h = data[AtomicDataDict.NODE_ATTRS_KEY] # just 1hot enc of atom types for now; make this optional
    #     assert x.shape[0] == h.shape[0]
    #     device = data['batch'].device
    #     bs = len(torch.unique(data['batch']))

    #     # Sample timestep t for batch
    #     debug = True
    #     if debug:
    #         t_int = self.cyclical_range_T(device)
    #     else:
    #         # T+1 since exclusive, shape: (bs, 1)
    #         t_int = torch.randint(0, self.T + 1, size=(bs, 1), device=device).float()

    #     # this is required in loss
    #     data['t_is_zero_mask'] = (t_int[data['batch']] == 0).float()  # used to mask L0 in loss calc (i.e. optimize log p(x | z0) iff t==0)
    #     t_embedding = self.get_time_embedding(t_int[data['batch']]) # "broadcast" the mol-wise t to each atom

    #     # Normalize t to [0, 1].
    #     t = t_int / self.T
    #     # Compute alpha_t and sigma_t from gamma as proposed in Variational Diffusion Models ppr eq (3) and (4)
    #     gamma_t = self.gamma(t) # gamma = noise scheduler
    #     alpha_t = alpha(gamma_t) # for each obs we have the associated val of alpha_t and sigma_t wrt the sampled t for that obs, shape: ([bs,1,1,])
    #     sigma_t = sigma(gamma_t)

    #     # fetch from scheduler the value for t=0
    #     # required in loss
    #     t_zeros = torch.zeros_like(t)
    #     gamma_0 = self.gamma(t_zeros)
    #     data['gamma_0'] = gamma_0[data['batch']]

    #     eps = self.sample_combined_position_feature_noise(x.shape, h.shape, device)
    #     data['noise_target'] = eps

    #     xh = torch.cat([x, h], dim=-1) # cat node feats to apply noisification on them in a single step

    #     # Sample zt ~ Normal(alpha_t x, sigma_t)
    #     # This projection only works because Gaussian is rotation invariant around
    #     # zero and samples are independent!
    #     # Sample z_t given x, h for timestep t, from q(z_t | x, h)
    #     alpha_t = alpha_t[data['batch']]
    #     sigma_t = sigma_t[data['batch']]
    #     z_t = alpha_t * xh + sigma_t * eps # noisified input
    #     #! diffusion_utils.assert_mean_zero_with_mask(z_t[:, :, :n_dims], node_mask)

    #     # Split z_t back into position and node features
    #     z_t_pos = z_t[:, :3]  # First 3 dimensions are position
    #     z_t_h = z_t[:, 3:]    # Remaining dimensions are node features

    #     # Update data dictionary with diffused values
    #     data[AtomicDataDict.POSITIONS_KEY] = z_t_pos
    #     assert h.shape[-1] == z_t_h.shape[-1]
    #     assert x.shape[-1] == z_t_pos.shape[-1] == 3
    #     assert x.shape[0] == z_t_h.shape[0]

    #     z_t_h_with_t_emb = torch.cat([z_t_h, t_embedding], dim=-1) # the most trivial t conditioning via cat
    #     data[AtomicDataDict.NODE_ATTRS_KEY] = z_t_h_with_t_emb
    #     data[AtomicDataDict.NODE_FEATURES_KEY] = z_t_h_with_t_emb

    #     return data

    # def OGforward_sampling(self, data:AtomicDataDict):
    #     # todo refactor to aggregate code from forward
    #     h = data[AtomicDataDict.NODE_ATTRS_KEY] # just 1hot enc of atom types for now; make this optional
    #     t_int = data['t_inference']
    #     t_embedding = self.get_time_embedding(t_int[data['batch']]) # "broadcast" the mol-wise t to each atom
    #     z_t_h_with_t_emb = torch.cat([h, t_embedding.squeeze()], dim=-1) # the most trivial t conditioning via cat
    #     data[AtomicDataDict.NODE_ATTRS_KEY] = z_t_h_with_t_emb
    #     data[AtomicDataDict.NODE_FEATURES_KEY] = z_t_h_with_t_emb
    #     return data

    # def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor):
    #     """
    #     Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

    #     These are defined as:
    #         alpha t given s = alpha t / alpha s,
    #         sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
    #     """
    #     sigma2_t_given_s = -expm1(softplus(gamma_s) - softplus(gamma_t))

    #     # alpha_t_given_s = alpha_t / alpha_s
    #     log_alpha2_t = F.logsigmoid(-gamma_t)
    #     log_alpha2_s = F.logsigmoid(-gamma_s)
    #     log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

    #     alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
    #     sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

    #     return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    # def sample_p_zs_given_zt(
    #     self,
    #     s,
    #     t,
    #     z_x_t,
    #     z_h_t,
    #     model,
    #     context,
    #     fix_noise=False):
    #     """Samples from zs ~ p(zs | zt). Only used during sampling."""
    #     gamma_s = self.gamma(s)
    #     gamma_t = self.gamma(t)

    #     sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s)

    #     sigma_s = sigma(gamma_s)
    #     sigma_t = sigma(gamma_t)


    #     # Get Neural net prediction.
    #     #! EDGE_VECTORS_KEY, EDGE_LENGTH_KEY are always recomputed inside the fwd of model
    #     args = {
    #         'pos':                              z_x_t,
    #         AtomicDataDict.NODE_ATTRS_KEY:      z_h_t,
    #         AtomicDataDict.NODE_FEATURES_KEY:   z_h_t,
    #         'r_max':                            torch.finfo(torch.float32).max,
    #         'batch':                            torch.zeros((z_x_t.shape[0], 1), device=z_x_t.device),
    #         't_inference':                      t,
    #     }
    #     data = AtomicData.from_points(**args)
    #     zt = torch.cat([z_x_t, z_h_t], dim=-1)
    #     eps_t = model(AtomicData.to_AtomicDataDict(data))

    #     # Compute mu for p(zs | zt).
    #     # Note: mu_{t->s} = 1 / alpha_{t|s} z_t - sigma_{t|s}^2 / sigma_t / alpha_{t|s} epsilon
    #     # follows from the definition of mu_{t->s} and Equ. (7) in the EDM paper:
    #     # replace the x in the definition of mu_{t->s} with x_hat as def in eq (7), after calculations you end with the formula for mu
    #     mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t['noise']

    #     # Compute sigma for p(zs | zt).
    #     _sigma = sigma_t_given_s * sigma_s / sigma_t # this is sigma_(t->s)

    #     # Sample zs given the paramters derived from zt.
    #     zs = self.sample_normal(mu, _sigma, fix_noise) # sampling of next time step

    #     # Project down to avoid numerical runaway of the center of gravity. i.e. recenter
    #     z_x = center_pos(zs[..., :3])
    #     z_h = zs[..., 3:]
    #     return z_x, z_h

    # def compute_x_pred(self, net_out, zt, gamma_t):
    #     """Commputes x_pred, i.e. the most likely prediction of x."""
    #     sigma_t = sigma(gamma_t)
    #     alpha_t = alpha(gamma_t)
    #     eps_t = net_out
    #     x_pred = 1. / alpha_t * (zt - sigma_t * eps_t)
    #     return x_pred

    # def SNR(self, gamma):
    #     """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
    #     return torch.exp(-gamma)

    # def sample_p_xh_given_z0(
    #     self,
    #     z_x_0,
    #     z_h_0,
    #     model,
    #     context,
    #     fix_noise=False):
    #     """Samples x ~ p(x|z0)."""
    #     zeros = torch.zeros(size=(z_x_0.size(0), 1), device=z_h_0.device) # t
    #     gamma_0 = self.gamma(zeros)

    #     # get noise
    #     args = {
    #         'pos':                              z_x_0,
    #         AtomicDataDict.NODE_ATTRS_KEY:      z_h_0,
    #         AtomicDataDict.NODE_FEATURES_KEY:   z_h_0,
    #         'r_max':                            torch.finfo(torch.float32).max,
    #         'batch':                            torch.zeros((z_x_0.shape[0], 1), device=z_h_0.device),
    #         't_inference':                      zeros,
    #     }
    #     data = AtomicData.from_points(**args)
    #     z0 = torch.cat([z_x_0, z_h_0], dim=-1)
    #     net_out = model(AtomicData.to_AtomicDataDict(data))

    #     # Compute mu for p(zs | zt).
    #     mu_x = self.compute_x_pred(net_out['noise'], z0, gamma_0)

    #     # Computes sqrt(sigma_0^2 / alpha_0^2)
    #     sigma_x = self.SNR(-0.5 * gamma_0)#.unsqueeze(1) # this sigma should be 0 right?
    #     xh = self.sample_normal(mu=mu_x, sigma=sigma_x, fix_noise=fix_noise)

    #     # ok so in the end is argmax for 1hot and torch.round for integers, it's just in the loss that u cannot cross entropy and mse
    #     x, h_cat = xh[..., :3], xh[..., 3:]
    #     h_cat = 4*h_cat # unscale

    #     # h_cat = F.one_hot(torch.argmax(h_cat, dim=2), 22)
    #     atom_numbers = torch.argmax(h_cat, dim=-1)
    #     return x, atom_numbers # return x, h_cat

    # def sample_normal(self, mu, sigma, fix_noise=False):
    #     """Samples from a Normal distribution."""
    #     N = mu.shape[0]
    #     eps = self.sample_combined_position_feature_noise((N,3), (N, mu.shape[-1]-3), mu.device)
    #     return mu + sigma * eps

    # def sample_combined_position_feature_noise(self, x_shape, h_shape, device):
    #     # TODO input args questionable could be (N, feat_dim) +
    #     #   check if u can get device from shape instead of requiring it as input
    #     """
    #     Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
    #     Used to initialize random mess at origin of axis during sampling
    #     return shape: (x_shape[0], x_shape[-1]+h_shape[-1])
    #     """
    #     z_x = center_pos((sample_noise_from_N_0_1(size=x_shape, device=device)))
    #     z_h = sample_noise_from_N_0_1(size=h_shape, device=device)
    #     return torch.cat([z_x, z_h], dim=-1)
###########
    # def cyclical_range_T(self, bs, device):
    #     # TODO: fix bs issue (still present in cyclical_range_T: it outs at last batch a bs-1 tensor of ts so index breaks down with cuda assert)
    #     # for single mol learning
    #     if not hasattr(self, 't_counter'):
    #         self.t_counter = 0

    #     offset = bs # this should be equal to bs

    #     # Get next offset timesteps in cyclic manner
    #     start_idx = self.t_counter
    #     end_idx = start_idx + offset

    #     if end_idx > self.T:
    #         end_idx = self.T

    #     t_int = torch.arange(start_idx, end_idx, device=device).float().unsqueeze(-1)

    #     # Update counter for next batch
    #     self.t_counter += offset
    #     if self.t_counter >= self.T:
    #         self.t_counter = 0

    #     return t_int