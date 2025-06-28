import torch
import numpy as np
from typing import Dict, List, Set, Tuple

class AtomTypeMapper(torch.nn.Module):
    """Utility class to map between atomic numbers and sequential indices for one-hot encoding."""

    def __init__(self, atomic_numbers: Set[int]):
        """
        Initialize the mapper with a set of atomic numbers.

        Args:
            atomic_numbers: Set of atomic numbers present in your dataset
        """
        super().__init__()
        # Sort atomic numbers to ensure consistent mapping
        self.atomic_numbers = sorted(list(atomic_numbers))
        # Create bidirectional mappings
        self._atom_to_idx: Dict[int, int] = {atom: idx for idx, atom in enumerate(self.atomic_numbers)}
        self._idx_to_atom: Dict[int, int] = {idx: atom for idx, atom in enumerate(self.atomic_numbers)}
        self.num_types = len(self.atomic_numbers)

    def atom_to_idx(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        """Convert atomic numbers to sequential indices."""
        return torch.tensor([self._atom_to_idx[atom.item()] for atom in atomic_numbers],
                          device=atomic_numbers.device)

    def idx_to_atom(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert sequential indices back to atomic numbers."""
        return torch.tensor([self._idx_to_atom[idx.item()] for idx in indices],
                          device=indices.device)

    def to_one_hot(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        """Convert atomic numbers to one-hot encoding."""
        indices = self.atom_to_idx(atomic_numbers)
        return torch.nn.functional.one_hot(indices, num_classes=self.num_types)

    def from_one_hot(self, one_hot: torch.Tensor) -> torch.Tensor:
        """Convert one-hot encoding back to atomic numbers."""
        indices = torch.argmax(one_hot, dim=-1)
        return self.idx_to_atom(indices)

    def get_num_types(self) -> int:
        """Get the number of unique atom types."""
        return self.num_types

    def get_atomic_numbers(self) -> List[int]:
        """Get the list of atomic numbers in order."""
        return self.atomic_numbers.copy()