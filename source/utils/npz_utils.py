import os
from types import SimpleNamespace
from typing import List, Union
import numpy as np


def ls(dir):
    if not os.path.isdir(dir):
        raise ValueError(f"dir provided ({dir}) is not a dir")
        # Efficiently pre-filter entries using os.scandir
    with os.scandir(dir) as entries:
        # Filter non-hidden files using entry.is_file() and entry.name
        return [os.path.join(dir, entry.name) for entry in entries if entry.is_file() and not entry.name.startswith('.')]

def get_field_from_npzs(path:str, field:Union[str, List]='*'):
  '''
  example usage:
  l = get_field_from_npzs(p)
  l[0][k] -> access to content
  '''
  is_single_npz = lambda path: os.path.splitext(path)[1].lower() == ".npz"
  npz_files = [path] if is_single_npz(path) else ls(path)
  if field == '*':
    return [np.load(npz) for npz in npz_files]
  possible_keys = (k for k in np.load(npz_files[0]).keys())
  if field not in possible_keys:
    raise ValueError(f'{field} not in {list(possible_keys)}')
  if isinstance(field, str):
    return [np.load(el)[field].item() for el in npz_files]
  if not isinstance(field, List):
    raise ValueError(f'Unaccepted type for field, which is {type(field)}, but should be List or str ')

  out = []
  for npz in ls(path):
    data = np.load(npz)
    sn  = SimpleNamespace()
    for f in field:
        sn.__setattr__(f, data[f].item())
    sn.__setattr__("path", path)
    out.append(sn)
  return out