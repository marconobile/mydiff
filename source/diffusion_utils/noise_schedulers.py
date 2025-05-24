import numpy as np
import torch

class NoiseScheduler(object):
    def __init__(self, scheduler, T:int):
        self.T = T

        if scheduler == "DDPM":
            self.alpha_bar, alphas, betas = linear_schedule(self.T, beta_minmax=[1e-4, 2e-2])
        elif scheduler == "poly":
            raise ValueError(f"Not ready")
            # self.alpha_bar = polynomial_schedule(self.T)
        elif scheduler == "iDDPM":
            self.alpha_bar, alphas, betas = cosine_beta_schedule(self.T)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")

        alphas = torch.tensor(alphas, dtype=torch.float32)
        betas = torch.tensor(betas, dtype=torch.float32)

        self.sqrt_betas = torch.sqrt(betas) # for inference
        self.one_minus_alphas = 1 - alphas # for inference
        self.sqrt_alphas = torch.sqrt(alphas) # for inference
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar) # for training
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1-self.alpha_bar) # for both

    def __call__(self, t, batch_idx, device):

        if self.sqrt_alpha_bar.device != device:
            self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)

        if self.sqrt_one_minus_alpha_bar.device != device:
            self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)

        alpha = self.sqrt_alpha_bar[t][batch_idx]
        sigma = self.sqrt_one_minus_alpha_bar[t][batch_idx]
        return alpha, sigma


def linear_schedule(Tmax, beta_minmax=[1e-4, 2e-2]):
    beta_1, beta_T = beta_minmax
    betas = torch.linspace(start=beta_1, end=beta_T, steps=Tmax)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return alpha_bars, alphas, betas

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

    return torch.tensor(alphas_cumprod, dtype=torch.float32), alphas, betas

# def clip_noise_schedule(alphas2, clip_value=0.001):
#     """
#     For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
#     sampling.
#     """
#     alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

#     alphas_step = (alphas2[1:] / alphas2[:-1])

#     alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
#     alphas2 = np.cumprod(alphas_step, axis=0)

#     return alphas2

# def polynomial_schedule(timesteps: int, s=1e-5, power=3.):
#     """
#     A noise schedule based on a simple polynomial equation: 1 - x^power.
#     """
#     steps = timesteps + 1
#     x = np.linspace(0, steps, steps)
#     alphas2 = (1 - np.power(x / steps, power))**2

#     alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

#     precision = 1 - 2 * s

#     alphas2 = precision * alphas2 + s

#     return torch.tensor(alphas2, dtype=torch.float32)





########################################################################

# class PredefinedNoiseSchedule(torch.nn.Module):
#     """
#     Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
#     self.gamma = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps_
#     """
#     def __init__(self, noise_schedule:str='polynomial_2', timesteps:int=1000, precision:float=1e-5):
#         super(PredefinedNoiseSchedule, self).__init__()
#         self.timesteps = timesteps

#         if noise_schedule == 'cosine':
#             alphas2 = cosine_beta_schedule(timesteps)
#         elif 'polynomial' in noise_schedule:
#             splits = noise_schedule.split('_')
#             assert len(splits) == 2
#             power = float(splits[1])
#             alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
#         else:
#             raise ValueError(noise_schedule)

#         # print('alphas2', alphas2)

#         sigmas2 = 1 - alphas2

#         log_alphas2 = np.log(alphas2)
#         log_sigmas2 = np.log(sigmas2)

#         log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

#         # print('gamma', -log_alphas2_to_sigmas2)

#         self.gamma = torch.nn.Parameter(
#             torch.from_numpy(-log_alphas2_to_sigmas2).float(),
#             requires_grad=False)

#     def forward(self, t):
#         assert torch.all(t >= 0) and torch.all(t <= 1), f"t must be between 0 and 1, got {t}"
#         t_int = torch.round(t * self.timesteps).long()
#         return self.gamma[t_int]


