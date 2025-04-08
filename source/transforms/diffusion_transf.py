import torch
import numpy as np
from copy import deepcopy

from geqtrain.data.AtomicData import AtomicData


def coord_noise(data:AtomicData, noise_scale:float=0.04):
    data = deepcopy(data) #! in-place op to the data obj persist thru dloader iterations
    data.noise_target = torch.from_numpy(np.random.normal(0, 1, size=data.pos.shape) * noise_scale).to(torch.float32)
    data.pos += data.noise_target
    return data