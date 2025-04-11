import torch
import torch.nn
import math
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from geqtrain.data import AtomicDataDict
from geqtrain.nn._graph_mixin import GraphModuleMixin

from typing import Dict, Optional, List


from source.utils import AtomTypeMapper

@compile_mode("script")
class OneHotAtomEncodingFromAtomNum(GraphModuleMixin, torch.nn.Module):

    num_types: int
    set_features: bool

    def __init__(
        self,
        num_types: int,
        set_features: bool = True,
        irreps_in=None,
        scaling_factor:float|None=None,
    ):
        super().__init__()
        self.num_types = num_types
        self.set_features = set_features
        self.scaling_factor = scaling_factor

        atomic_numbers = {1, 2, 4, 5, 6, 7, 8, 9, 12, 14, 15, 16, 17, 18, 20, 22, 30, 31, 32, 33, 34, 35} # Your unique atomic numbers in dataset
        self.atomnum2onehot = AtomTypeMapper(atomic_numbers)

        # Output irreps are num_types even (invariant) scalars
        irreps_out = {AtomicDataDict.NODE_ATTRS_KEY: Irreps([(self.num_types, (0, 1))])}
        if self.set_features:
            irreps_out[AtomicDataDict.NODE_FEATURES_KEY] = irreps_out[AtomicDataDict.NODE_ATTRS_KEY]
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out) # my guess: None -> NatomsTypes of l = 0, defines inpt/outpt shapes

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if data.get(AtomicDataDict.NODE_ATTRS_KEY, None) is None:
            type_numbers = data.get(AtomicDataDict.NODE_TYPE_KEY).squeeze(-1)

            #! bug in my atomic number encoding
            with torch.no_grad():
                type_numbers = type_numbers + 1.0

            one_hot = self.atomnum2onehot.to_one_hot(type_numbers).to(device=type_numbers.device, dtype=data[AtomicDataDict.POSITIONS_KEY].dtype) # my guess: [bs, n, NatomsTypes]
            assert torch.all(torch.sum(one_hot, dim=-1) == 1.0)
            if self.scaling_factor is not None:
                one_hot = self.scaling_factor*one_hot
                assert torch.all(torch.sum(one_hot, dim=-1) == self.scaling_factor)

            data[AtomicDataDict.NODE_ATTRS_KEY] = one_hot
            data['scaled_node_types_one_hot'] = one_hot
            if self.set_features:
                data[AtomicDataDict.NODE_FEATURES_KEY] = one_hot
        return data