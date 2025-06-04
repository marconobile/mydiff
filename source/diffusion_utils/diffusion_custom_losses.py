import torch
import torch.nn as nn
from torch_scatter import scatter_mean
import os

class T_MSE:
    def __init__(
        self,
        func_name: str='T_MSE',
        params: dict = {},
        **kwargs,
    ):
        self.func_name = func_name
        self.mse = nn.MSELoss(reduction='none') # # reduction is by default mean
        for key, value in kwargs.items():
            setattr(self, key, value)
        for key, value in params.items():
            setattr(self, key, value)

        self.log_file = os.path.join(self.save_path, "loss_per_t.txt")

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
        **kwargs,
    ):
        preds = pred[key]
        targets = ref[key]
        losses = self.mse(targets, preds)
        ts = pred['sampled_t'].squeeze()

        out = torch.zeros(size=(self.tmax,), device=preds.device)
        out = scatter_mean(src = losses.sum(-1), index=ts, out=out)
        out[out == 0.0] = float('nan')
        with open(self.log_file, "a") as f:
            f.write(f"{out.tolist()}\n")
        return losses


class C_MSE:
    def __init__(
        self,
        func_name: str='C_MSE',
        params: dict = {},
        **kwargs,
    ):
        self.func_name = func_name
        self.mse = nn.MSELoss(reduction='none') # # reduction is by default mean
        for key, value in kwargs.items():
            setattr(self, key, value)
        for key, value in params.items():
            setattr(self, key, value)

        self.log_file = os.path.join(self.save_path, "loss_per_c.txt")
        self.fill_value = -1.0

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
        **kwargs,
    ):
        preds = pred[key]
        targets = ref[key]
        batch = pred['batch'].squeeze()    # (N,)
        losses = self.mse(targets, preds)
        # First, mean over the 3 dimensions for each node
        node_loss = losses.mean(dim=1)     # (N,)
        # Then, aggregate by batch index (G groups)
        losses = scatter_mean(node_loss, batch, dim=0)  # (G,)
        labels = pred['sampled_labels'].squeeze()
        out = torch.full(size=(self.number_of_labels,), fill_value=self.fill_value, device=preds.device)
        out = scatter_mean(losses, labels, dim=0, out=out)  # (G,)
        out[out == self.fill_value] = float('nan')
        with open(self.log_file, "a") as f:
            f.write(f"{out.tolist()}\n")
        return losses



# class DiffusionLoss:
#     def __init__(
#         self,
#         func_name: str='DiffusionLoss',
#         params: dict = {},
#         **kwargs,
#     ):
#         self.func_name = 'DiffusionLoss'
#         self.params = params
#         self.mse = nn.MSELoss(reduction='none')
#         for key, value in kwargs.items():
#             setattr(self, key, value)

#     def log_pxh_given_z0_without_constants(self, preds, targets, epsilon=1e-10):
#         '''
#         https://drive.google.com/file/d/10Ix3lyyMLojnxlIJSKknpW4n6D7lwbtg/view?usp=sharing,
#         https://aistudio.google.com/app/prompts?state=%7B%22ids%22:%5B%221AUEo5pRYT6-IvNG8ylnugYyGwUTIO3yE%22%5D,%22action%22:%22open%22,%22userId%22:%22115197647800952079255%22,%22resourceKeys%22:%7B%7D%7D&usp=sharing

#         From paper:
#         Following (Ho et al., 2020) during training we set w(t) = 1 as it stabilizes training and it is known to improve sample quality for images.
#         Experimentally we also found this to hold true for molecules: even when evaluating the probabilistic variational objective for which w(t) = (1−SNR(t−1)/SNR(t)),
#         the model trained with w(t) = 1 outperformed models trained with the variational w(t).
#         So this is what happens for eq (17) when not using l2 loss (but used when vlb? or when gamma is learnt?)
#         In the paper they also say that log(Z**-1) is also added else where, will it be added in the sampling?
#         '''
#         n_dims = 3
#         one_hot_scaling_factor = 4
#         noise_hat = preds['noise']
#         noise_target = targets['noise']
#         gamma_0 = preds['gamma_0']

#         # L0 for continuous pos:
#         # Computes the error for the distribution N(x | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0),
#         # the weighting in the epsilon parametrization is exactly '1'.
#         # L_0^(x) below eq (19) in the paper, -.5 since w(t=0) = −1; nb w(t) = (1−SNR(t−1)/SNR(t)) and it must be DEFD for t=0! since what would be SNR(0−1)!?
#         eps_x = noise_hat[:, :n_dims]
#         net_x = noise_target[:, :n_dims]
#         log_p_x_given_z_without_normalization_constant_Z = -0.5 * torch.mean(self.mse(net_x, eps_x), dim=-1) #self.mse(net_x, eps_x).sum(-1)/n_dims # todo use .mean( here

#         # L0 for categorical features:
#         # step 1: Compute sigma_0 and rescale to the integer scale of the data
#         sigma_0 = sigma(gamma_0)
#         sigma_0_cat = sigma_0 * one_hot_scaling_factor

#         # step 2.1: compute log p(h_cat | z_0)
#         # Extract from z_t the atom categories
#         z_h_cat = noise_hat[:, n_dims:]  # shape torch.Size([bs, maxN, natomcategories])
#         estimated_h_cat = z_h_cat * one_hot_scaling_factor # undo the scaling of the 1hot also in the prediction
#         onehot_target = preds['scaled_node_types_one_hot'] * one_hot_scaling_factor # undo the scaling of the 1hot; nb key not present in targets
#         # Centered h_cat around 1, since onehot encoded.
#         centered_h_cat = estimated_h_cat - 1

#         # Compute integrals from 0.5 to 1.5 of the normal distribution
#         # N(mean=z_h_cat, stdev=sigma_0_cat)
#         # Goal: compute the probability (so a val in [0,1]) that the original integer is target h_integer (i.e. p(h_integer | z_h_int))
#         # How? -> Compute integral from -0.5 to 0.5 of N(mean=h_integer_centered, stdev=sigma_0_int) and use it as a proxy measure of likelihood (i.e. computes the probability that predicted_integer falls in +/-.5 around target_integer)
#         # normal dist AUC based version of Cross Entropy: if predicted_integer is far from the target_integer then log_ph_integer is low, high if viceversa; (easier to viz in PDF terms then CDF)
#         log_ph_cat_proportional = torch.log(
#             cdf_standard_gaussian((centered_h_cat + 0.5) / sigma_0_cat) # if sigma_0_int high -> higher variance -> less precision required ; if sigma_0_int small -> peaked gauss -> higher precision required
#             - cdf_standard_gaussian((centered_h_cat - 0.5) / sigma_0_cat) # if t almost 0 then we want high precision
#             + epsilon)

#         # Normalize the distribution over the categories.
#         log_Z = torch.logsumexp(log_ph_cat_proportional, dim=-1, keepdim=True)
#         log_probabilities = log_ph_cat_proportional - log_Z

#         # Select the log_prob of the current category usign the onehot representation
#         log_p_h_given_z = log_probabilities * onehot_target
#         log_p_h_given_z = log_p_h_given_z.sum(-1)

#         return log_p_x_given_z_without_normalization_constant_Z + log_p_h_given_z

#     def __call__(
#         self,
#         pred: dict,
#         ref: dict,
#         key: str,
#         mean: bool = True,
#         **kwargs,
#     ):
#         '''
#         Skipping computation of: self.kl_prior(xh, node_mask)
#         The KL between q(z1 | x) and p(z1) = Normal(0, 1). Should be close to zero.
#         From their code: 'compute it so that you see it when you've made a mistake in your noise schedule.'
#         it is just to check and "ensure" that u diffuse into N(0,1), but I can already check it by checking the vals via debug
#         '''
#         predicted_noise = pred[key] # shape: (N, 3+1hot_categories)
#         target_noise = ref[key] # shape: (N, 3+1hot_categories)

#         # equation (17) in Equivariant Diffusion for Molecule Generation in 3D ppr;
#         # SNR_weight all set to 1 as said in ppr
#         # the loss below must can be aggregated over dim=-1 but must be kept distinctinct among atoms, since it must have same shape of t_is_zero mask
#         loss_t_larger_than_zero = 0.5 * torch.mean(self.mse(predicted_noise, target_noise), dim=-1) #.sum(-1)/target_noise.shape[-1]

#         # Computes the L_0 term (even if gamma_t is not actually gamma_0)
#         loss_for_when_t_is_0 = -self.log_pxh_given_z0_without_constants(pred, ref)

#         t_is_zero = pred['t_is_zero_mask']
#         t_is_not_zero = 1 - t_is_zero
#         loss = loss_for_when_t_is_0 * t_is_zero.squeeze() + t_is_not_zero.squeeze() * loss_t_larger_than_zero

#         if mean: return torch.mean(loss)
#         return loss

