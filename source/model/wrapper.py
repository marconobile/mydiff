# import logging
# from typing import List, Optional

# from geqtrain.data import AtomicDataDict
# from geqtrain.nn import (
#     SequentialGraphNetwork,
#     ReadoutModule,
#     # GVPGeqTrain,
#     WeightedTP,
#     TransformerBlock,
# )

# from geqtrain.utils import Config
# from geqtrain.data import AtomicDataDict
# from torch.utils.data import ConcatDataset

# from source.diffusion_utils.scheduler import ForwardDiffusionModule
# from source.nn.atomNum21hot import OneHotAtomEncodingFromAtomNum


# def DiffusionWrapper(model, config: Config, initialize: bool, dataset: Optional[ConcatDataset] = None) -> SequentialGraphNetwork:
#     '''
#     instanciates a layer with multiple ReadoutModules
#     '''

#     logging.info("--- Building ForwardDiffusionModule Module ---")

#     if 'node_attributes' in config:
#         node_embedder = (OneHotAtomEncodingFromAtomNum, dict(
#             out_field=AtomicDataDict.NODE_ATTRS_KEY,
#             attributes=config.get('node_attributes'),
#             scaling_factor=config.get('one_hot_scaling_factor', None),
#         ))
#     else:
#         raise ValueError('Missing node_attributes in yaml')

#     layers = {
#         "node_attrs": node_embedder,
#         "diffusion_module": ForwardDiffusionModule,
#     }

#     layers.update({
#         "wrapped_model": model,
#     })

#     return SequentialGraphNetwork.from_parameters(
#         shared_params=config,
#         layers=layers,
#     )