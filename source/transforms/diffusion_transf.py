from copy import deepcopy

from geqtrain.data.AtomicData import AtomicData
from source.diffusion_utils import center_pos


def center_pos_transf(data:AtomicData):
    data = deepcopy(data) #! in-place op to the data obj persist thru dloader iterations
    data.pos = center_pos(data.pos)
    return data