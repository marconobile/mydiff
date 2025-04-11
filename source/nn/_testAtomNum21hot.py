# Test with your specific atomic numbers

import torch
from source.utils import AtomTypeMapper

atomic_numbers = {0, 34, 4, 5, 6, 7, 8, 13, 14, 15, 16}
mapper = AtomTypeMapper(atomic_numbers)

# Test initialization
assert mapper.num_types == 11
assert len(mapper.get_atomic_numbers()) == 11

# Test conversion from atomic numbers to indices
test_atoms = torch.tensor([0, 34, 4, 5, 6, 7, 8, 13, 14, 15, 16])
indices = mapper.atom_to_idx(test_atoms)
assert len(indices) == 11
assert torch.all(indices < mapper.num_types)

# Test conversion back to atomic numbers
recovered_atoms = mapper.idx_to_atom(indices)
assert torch.all(test_atoms == recovered_atoms)

# Test one-hot encoding
one_hot = mapper.to_one_hot(test_atoms)
assert one_hot.shape == (11, 11)  # (num_atoms, num_types)
assert torch.all(one_hot.sum(dim=1) == 1)  # Each atom should have exactly one 1

# Test conversion back from one-hot
recovered_from_one_hot = mapper.from_one_hot(one_hot)
assert torch.all(test_atoms == recovered_from_one_hot)

# Test with a single atom
single_atom = torch.tensor([34])
single_one_hot = mapper.to_one_hot(single_atom)
assert single_one_hot.shape == (1, 11)
assert torch.all(single_one_hot.sum() == 1)

# Test device consistency
if torch.cuda.is_available():
    device = torch.device('cuda')
    test_atoms_gpu = test_atoms.to(device)
    one_hot_gpu = mapper.to_one_hot(test_atoms_gpu)
    assert one_hot_gpu.device.type == device.type
    recovered_gpu = mapper.from_one_hot(one_hot_gpu)
    assert recovered_gpu.device.type == device.type
    assert torch.all(test_atoms_gpu == recovered_gpu)

atomic_numbers = {0, 34, 4, 5, 6, 7, 8, 13, 14, 15, 16}  # Your unique atomic numbers in dataset
atom_mapper = AtomTypeMapper(atomic_numbers)

atomic_numbers_tensor = torch.tensor([0, 34, 4, 5, 6, 7, 8, 13, 14, 15, 16])
one_hot = atom_mapper.to_one_hot(atomic_numbers_tensor)