# Taken from https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/equivariant_diffusion/en_diffusion.py

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Union, List
from e3nn import o3
from torch.nn import functional as F
from geqtrain.data import AtomicData, AtomicDataDict


from source.diffusion_utils.utils import (
    sample_noise_from_N_0_1,
    cdf_standard_gaussian,
    sigma,
    alpha,
    center_pos,
)
from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin

def expm1(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x)


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)

def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power))**2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    self.gamma = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps_
    """
    def __init__(self, noise_schedule:str='polynomial_2', timesteps:int=1000, precision:float=1e-5):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            alphas2 = cosine_beta_schedule(timesteps)
        elif 'polynomial' in noise_schedule:
            splits = noise_schedule.split('_')
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        else:
            raise ValueError(noise_schedule)

        print('alphas2', alphas2)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        print('gamma', -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)

    def forward(self, t):
        assert torch.all(t >= 0) and torch.all(t <= 1), f"t must be between 0 and 1, got {t}"
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]


class ForwardDiffusionModule(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        irreps_in:dict={},
        out_field: Optional[str] = 'noise',
    ):
        super().__init__()
        self.out_field = out_field

        self.gamma = PredefinedNoiseSchedule() # naming inherited from ref for code compatibility
        self.T = self.gamma.timesteps
        self.ref_data_keys = ['noise_target']
        self.t_embedding_dim = 64 # 64 sin and 64 cos

        out_irrep = o3.Irreps([(mul+2*self.t_embedding_dim, ir) for mul, ir in irreps_in[AtomicDataDict.NODE_ATTRS_KEY]])

        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={
                AtomicDataDict.NODE_ATTRS_KEY:out_irrep,
                AtomicDataDict.NODE_FEATURES_KEY:out_irrep
                },
        )

    def get_time_embedding(self, timestep:int):
        # cast int to vector
        # same as transformer for positional embedding
        # (self.t_embedding_dim,)
        freqs = torch.pow(10000, -torch.arange(0, self.t_embedding_dim, dtype=torch.float32, device=timestep.device)/self.t_embedding_dim)
        # (1, self.t_embedding_dim)
        x = timestep* freqs
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

    def sample_normal(self, mu, sigma, fix_noise=False):
        """Samples from a Normal distribution."""
        N = mu.shape[0]
        eps = self.sample_combined_position_feature_noise((N,3), (N, mu.shape[-1]-3), mu.device)
        return mu + sigma * eps

    def sample_combined_position_feature_noise(self, x_shape, h_shape, device):
        # TODO input args questionable could be (N, feat_dim) +
        #   check if u can get device from shape instead of requiring it as input
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        Used to initialize random mess at origin of axis during sampling
        return shape: (x_shape[0], x_shape[-1]+h_shape[-1])
        """
        z_x = center_pos((sample_noise_from_N_0_1(size=x_shape, device=device)))
        z_h = sample_noise_from_N_0_1(size=h_shape, device=device)
        return torch.cat([z_x, z_h], dim=-1)

    def forward(self, data:AtomicDataDict):

        if 't_inference' in data:
            # can't use the normal forward since scheduler params are different for sampling transitions
            return self.forward_sampling(data)

        bs = len(torch.unique(data['batch']))
        _device = data['batch'].device

        x = data['pos'] # option: drop centering transform and center data['pos'] here considering batch
        h = data[AtomicDataDict.NODE_ATTRS_KEY] # just 1hot enc of atom types for now; make this optional
        assert x.shape[0] == h.shape[0]

        # Sample timestep t for batch
        # T+1 since exclusive, shape: (bs, 1)
        # t_int = torch.randint(0, self.T + 1, size=(bs, 1), device=_device).float()

        ################## for single mol learning
        if not hasattr(self, 't_counter'):
            self.t_counter = 0

        # Get next 100 timesteps in cyclic manner
        start_idx = self.t_counter
        end_idx = start_idx + 100
        if end_idx > self.T:
            end_idx = self.T

        t_int = torch.arange(start_idx, end_idx, device=_device).float().unsqueeze(-1)

        # Update counter for next batch
        self.t_counter += 100
        if self.t_counter >= self.T:
            self.t_counter = 0
        ################## for single mol learning

        # this is required in loss
        data['t_is_zero_mask'] = (t_int[data['batch']] == 0).float()  # used to mask L0 in loss calc (i.e. optimize log p(x | z0) iff t==0)
        t_embedding = self.get_time_embedding(t_int[data['batch']]) # "broadcast" the mol-wise t to each atom

        # Normalize t to [0, 1].
        t = t_int / self.T
        # Compute alpha_t and sigma_t from gamma as proposed in Variational Diffusion Models ppr eq (3) and (4)
        gamma_t = self.gamma(t) # gamma = noise scheduler
        alpha_t = alpha(gamma_t) # for each obs we have the associated val of alpha_t and sigma_t wrt the sampled t for that obs, shape: ([bs,1,1,])
        sigma_t = sigma(gamma_t)

        # fetch from scheduler the value for t=0
        # required in loss
        t_zeros = torch.zeros_like(t)
        gamma_0 = self.gamma(t_zeros)
        data['gamma_0'] = gamma_0[data['batch']]

        eps = self.sample_combined_position_feature_noise(x.shape, h.shape, _device)
        data['noise_target'] = eps

        xh = torch.cat([x, h], dim=-1) # cat node feats to apply noisification on them in a single step

        # Sample zt ~ Normal(alpha_t x, sigma_t)
        # This projection only works because Gaussian is rotation invariant around
        # zero and samples are independent!
        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        alpha_t = alpha_t[data['batch']]
        sigma_t = sigma_t[data['batch']]
        z_t = alpha_t * xh + sigma_t * eps # noisified input
        #! diffusion_utils.assert_mean_zero_with_mask(z_t[:, :, :n_dims], node_mask)

        # Split z_t back into position and node features
        z_t_pos = z_t[:, :3]  # First 3 dimensions are position
        z_t_h = z_t[:, 3:]    # Remaining dimensions are node features

        # Update data dictionary with diffused values
        data[AtomicDataDict.POSITIONS_KEY] = z_t_pos
        assert h.shape[-1] == z_t_h.shape[-1]
        assert x.shape[-1] == z_t_pos.shape[-1] == 3
        assert x.shape[0] == z_t_h.shape[0]

        z_t_h_with_t_emb = torch.cat([z_t_h, t_embedding], dim=-1) # the most trivial t conditioning via cat
        data[AtomicDataDict.NODE_ATTRS_KEY] = z_t_h_with_t_emb
        data[AtomicDataDict.NODE_FEATURES_KEY] = z_t_h_with_t_emb

        return data

    def forward_sampling(self, data:AtomicDataDict):
        # todo refactor to aggregate code from forward
        h = data[AtomicDataDict.NODE_ATTRS_KEY] # just 1hot enc of atom types for now; make this optional
        t_int = data['t_inference']
        t_embedding = self.get_time_embedding(t_int[data['batch']]) # "broadcast" the mol-wise t to each atom
        z_t_h_with_t_emb = torch.cat([h, t_embedding.squeeze()], dim=-1) # the most trivial t conditioning via cat
        data[AtomicDataDict.NODE_ATTRS_KEY] = z_t_h_with_t_emb
        data[AtomicDataDict.NODE_FEATURES_KEY] = z_t_h_with_t_emb
        return data

    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = -expm1(softplus(gamma_s) - softplus(gamma_t))

        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    def sample_p_zs_given_zt(
        self,
        s,
        t,
        z_x_t,
        z_h_t,
        model,
        context,
        fix_noise=False
    ):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s)

        sigma_s = sigma(gamma_s)
        sigma_t = sigma(gamma_t)


        # Get Neural net prediction.
        #! EDGE_VECTORS_KEY, EDGE_LENGTH_KEY are always recomputed inside the fwd of model
        args = {
            'pos':                              z_x_t,
            AtomicDataDict.NODE_ATTRS_KEY:      z_h_t,
            AtomicDataDict.NODE_FEATURES_KEY:   z_h_t,
            'r_max':                            torch.finfo(torch.float32).max,
            'batch':                            torch.zeros((z_x_t.shape[0], 1), device=z_x_t.device),
            't_inference':                      t,
        }
        data = AtomicData.from_points(**args)
        zt = torch.cat([z_x_t, z_h_t], dim=-1)
        eps_t = model(AtomicData.to_AtomicDataDict(data))

        # Compute mu for p(zs | zt).
        # Note: mu_{t->s} = 1 / alpha_{t|s} z_t - sigma_{t|s}^2 / sigma_t / alpha_{t|s} epsilon
        # follows from the definition of mu_{t->s} and Equ. (7) in the EDM paper:
        # replace the x in the definition of mu_{t->s} with x_hat as def in eq (7), after calculations you end with the formula for mu
        mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t['noise']

        # Compute sigma for p(zs | zt).
        _sigma = sigma_t_given_s * sigma_s / sigma_t # this is sigma_(t->s)

        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, _sigma, fix_noise) # sampling of next time step

        # Project down to avoid numerical runaway of the center of gravity. i.e. recenter
        z_x = center_pos(zs[..., :3])
        z_h = zs[..., 3:]
        return z_x, z_h

    def compute_x_pred(self, net_out, zt, gamma_t):
        """Commputes x_pred, i.e. the most likely prediction of x."""
        sigma_t = sigma(gamma_t)
        alpha_t = alpha(gamma_t)
        eps_t = net_out
        x_pred = 1. / alpha_t * (zt - sigma_t * eps_t)
        return x_pred

    def SNR(self, gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)

    def sample_p_xh_given_z0(
        self,
        z_x_0,
        z_h_0,
        model,
        context,
        fix_noise=False
    ):
        """Samples x ~ p(x|z0)."""
        zeros = torch.zeros(size=(z_x_0.size(0), 1), device=z_h_0.device) # t
        gamma_0 = self.gamma(zeros)

        # get noise
        args = {
            'pos':                              z_x_0,
            AtomicDataDict.NODE_ATTRS_KEY:      z_h_0,
            AtomicDataDict.NODE_FEATURES_KEY:   z_h_0,
            'r_max':                            torch.finfo(torch.float32).max,
            'batch':                            torch.zeros((z_x_0.shape[0], 1), device=z_h_0.device),
            't_inference':                      zeros,
        }
        data = AtomicData.from_points(**args)
        z0 = torch.cat([z_x_0, z_h_0], dim=-1)
        net_out = model(AtomicData.to_AtomicDataDict(data))

        # Compute mu for p(zs | zt).
        mu_x = self.compute_x_pred(net_out['noise'], z0, gamma_0)

        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0)#.unsqueeze(1) # this sigma should be 0 right?
        xh = self.sample_normal(mu=mu_x, sigma=sigma_x, fix_noise=fix_noise)

        # ok so in the end is argmax for 1hot and torch.round for integers, it's just in the loss that u cannot cross entropy and mse
        x, h_cat = xh[..., :3], xh[..., 3:]
        h_cat = 4*h_cat # unscale

        # h_cat = F.one_hot(torch.argmax(h_cat, dim=2), 22)
        atom_numbers = torch.argmax(h_cat, dim=-1)
        return x, atom_numbers # return x, h_cat


class DiffusionLoss:
    def __init__(
        self,
        func_name: str='DiffusionLoss',
        params: dict = {},
        **kwargs,
    ):
        self.func_name = 'DiffusionLoss'
        self.params = params
        self.mse = nn.MSELoss(reduction='none')
        for key, value in kwargs.items():
            setattr(self, key, value)

    def log_pxh_given_z0_without_constants(self, preds, targets, epsilon=1e-10):
        '''
        https://drive.google.com/file/d/10Ix3lyyMLojnxlIJSKknpW4n6D7lwbtg/view?usp=sharing,
        https://aistudio.google.com/app/prompts?state=%7B%22ids%22:%5B%221AUEo5pRYT6-IvNG8ylnugYyGwUTIO3yE%22%5D,%22action%22:%22open%22,%22userId%22:%22115197647800952079255%22,%22resourceKeys%22:%7B%7D%7D&usp=sharing

        From paper:
        Following (Ho et al., 2020) during training we set w(t) = 1 as it stabilizes training and it is known to improve sample quality for images.
        Experimentally we also found this to hold true for molecules: even when evaluating the probabilistic variational objective for which w(t) = (1−SNR(t−1)/SNR(t)),
        the model trained with w(t) = 1 outperformed models trained with the variational w(t).
        So this is what happens for eq (17) when not using l2 loss (but used when vlb? or when gamma is learnt?)
        In the paper they also say that log(Z**-1) is also added else where, will it be added in the sampling?
        '''
        n_dims = 3
        one_hot_scaling_factor = 4
        noise_hat = preds['noise']
        noise_target = targets['noise']
        gamma_0 = preds['gamma_0']

        # L0 for continuous pos:
        # Computes the error for the distribution N(x | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0),
        # the weighting in the epsilon parametrization is exactly '1'.
        # L_0^(x) below eq (19) in the paper, -.5 since w(t=0) = −1; nb w(t) = (1−SNR(t−1)/SNR(t)) and it must be DEFD for t=0! since what would be SNR(0−1)!?
        eps_x = noise_hat[:, :n_dims]
        net_x = noise_target[:, :n_dims]
        log_p_x_given_z_without_normalization_constant_Z = -0.5 * torch.mean(self.mse(net_x, eps_x), dim=-1) #self.mse(net_x, eps_x).sum(-1)/n_dims # todo use .mean( here

        # L0 for categorical features:
        # step 1: Compute sigma_0 and rescale to the integer scale of the data
        sigma_0 = sigma(gamma_0)
        sigma_0_cat = sigma_0 * one_hot_scaling_factor

        # step 2.1: compute log p(h_cat | z_0)
        # Extract from z_t the atom categories
        z_h_cat = noise_hat[:, n_dims:]  # shape torch.Size([bs, maxN, natomcategories])
        estimated_h_cat = z_h_cat * one_hot_scaling_factor # undo the scaling of the 1hot also in the prediction
        onehot_target = preds['scaled_node_types_one_hot'] * one_hot_scaling_factor # undo the scaling of the 1hot; nb key not present in targets
        # Centered h_cat around 1, since onehot encoded.
        centered_h_cat = estimated_h_cat - 1

        # Compute integrals from 0.5 to 1.5 of the normal distribution
        # N(mean=z_h_cat, stdev=sigma_0_cat)
        # Goal: compute the probability (so a val in [0,1]) that the original integer is target h_integer (i.e. p(h_integer | z_h_int))
        # How? -> Compute integral from -0.5 to 0.5 of N(mean=h_integer_centered, stdev=sigma_0_int) and use it as a proxy measure of likelihood (i.e. computes the probability that predicted_integer falls in +/-.5 around target_integer)
        # normal dist AUC based version of Cross Entropy: if predicted_integer is far from the target_integer then log_ph_integer is low, high if viceversa; (easier to viz in PDF terms then CDF)
        log_ph_cat_proportional = torch.log(
            cdf_standard_gaussian((centered_h_cat + 0.5) / sigma_0_cat) # if sigma_0_int high -> higher variance -> less precision required ; if sigma_0_int small -> peaked gauss -> higher precision required
            - cdf_standard_gaussian((centered_h_cat - 0.5) / sigma_0_cat) # if t almost 0 then we want high precision
            + epsilon)

        # Normalize the distribution over the categories.
        log_Z = torch.logsumexp(log_ph_cat_proportional, dim=-1, keepdim=True)
        log_probabilities = log_ph_cat_proportional - log_Z

        # Select the log_prob of the current category usign the onehot representation
        log_p_h_given_z = log_probabilities * onehot_target
        log_p_h_given_z = log_p_h_given_z.sum(-1)

        return log_p_x_given_z_without_normalization_constant_Z + log_p_h_given_z

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
        **kwargs,
    ):
        '''
        Skipping computation of: self.kl_prior(xh, node_mask)
        The KL between q(z1 | x) and p(z1) = Normal(0, 1). Should be close to zero.
        From their code: 'compute it so that you see it when you've made a mistake in your noise schedule.'
        it is just to check and "ensure" that u diffuse into N(0,1), but I can already check it by checking the vals via debug
        '''
        predicted_noise = pred[key] # shape: (N, 3+1hot_categories)
        target_noise = ref[key] # shape: (N, 3+1hot_categories)

        # equation (17) in Equivariant Diffusion for Molecule Generation in 3D ppr;
        # SNR_weight all set to 1 as said in ppr
        # the loss below must can be aggregated over dim=-1 but must be kept distinctinct among atoms, since it must have same shape of t_is_zero mask
        loss_t_larger_than_zero = 0.5 * torch.mean(self.mse(predicted_noise, target_noise), dim=-1) #.sum(-1)/target_noise.shape[-1]

        # Computes the L_0 term (even if gamma_t is not actually gamma_0)
        loss_for_when_t_is_0 = -self.log_pxh_given_z0_without_constants(pred, ref)

        t_is_zero = pred['t_is_zero_mask']
        t_is_not_zero = 1 - t_is_zero
        loss = loss_for_when_t_is_0 * t_is_zero.squeeze() + t_is_not_zero.squeeze() * loss_t_larger_than_zero

        if mean: return torch.mean(loss)
        return loss

