import torch

def get_t_embedder(t_embedder, t_embedding_dim, Tmax):
    if t_embedder == 'positional':
        return PositionalEmbedding(t_embedding_dim)
    elif t_embedder == 'fourier':
        raise ValueError(f"fourier embedder not ready")
        return FourierEmbedding(t_embedding_dim, Tmax)
    raise ValueError(f"Unknown t_embedder: {t_embedder}")


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, t_embedding_dim:int=128):
        super().__init__()
        self.t_embedding_dim = t_embedding_dim
        self.out_size = 2*self.t_embedding_dim

    def forward(self, t:int, batch_idx):
        # out is 2*t_embedding_dim: 1/2 sins and 1/2 cos
        # cast int to vector
        # same as transformer for positional embedding
        # (self.t_embedding_dim,)
        freqs = torch.pow(10000, -torch.arange(0, self.t_embedding_dim, dtype=torch.float32, device=t.device)/self.t_embedding_dim)
        # (1, self.t_embedding_dim)
        x = t* freqs
        emb = torch.cat([torch.cos(x), torch.sin(x)], dim=-1) # broadcast the mol-wise t to each atom
        return emb[batch_idx] # batch-ordered embeddings of t


# class FourierEmbedding(nn.Module):
#     """Fourier embedding layer."""

#     def __init__(self, dim, tmax):
#         """Initialize the Fourier Embeddings.

#         Parameters
#         ----------
#         dim : int
#             The dimension of the embeddings.

#         """

#         super().__init__()
#         self.proj = nn.Linear(1, dim)
#         self.proj_norm = nn.LayerNorm(dim)
#         torch.nn.init.normal_(self.proj.weight, mean=0, std=1)
#         torch.nn.init.normal_(self.proj.bias, mean=0, std=1)
#         self.proj.requires_grad_(False)
#         self.tmax = tmax

#     def forward(
#         self,
#         times,
#     ):
#         times = torch.log(times/self.tmax) * 0.25
#         # times = rearrange(times, "b -> b 1")
#         rand_proj = self.proj(times)
#         return self.proj_norm(torch.cos(2 * math.pi * rand_proj))
