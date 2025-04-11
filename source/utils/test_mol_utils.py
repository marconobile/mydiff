import unittest
import torch
from rdkit import Chem
from .mol_utils import coords_atomicnum_to_mol, coords_group_period_to_mol

class TestMolUtils(unittest.TestCase):
    def test_coords_atomicnum_to_mol(self):
        # Test with a simple molecule (methane)
        coords = torch.tensor([
            [0.0, 0.0, 0.0],  # C
            [1.0, 0.0, 0.0],  # H
            [0.0, 1.0, 0.0],  # H
            [0.0, 0.0, 1.0],  # H
            [-1.0, 0.0, 0.0],  # H
        ])
        atomic_nums = torch.tensor([6, 1, 1, 1, 1])  # C, H, H, H, H

        mol = coords_atomicnum_to_mol(coords, atomic_nums)
        self.assertEqual(mol.GetNumAtoms(), 5)
        self.assertEqual(mol.GetNumBonds(), 4)

        # Check that the molecule is methane
        smiles = Chem.MolToSmiles(mol)
        self.assertEqual(smiles, 'C')

    def test_coords_group_period_to_mol(self):
        # Test with a simple molecule (methane)
        coords = torch.tensor([
            [0.0, 0.0, 0.0],  # C
            [1.0, 0.0, 0.0],  # H
            [0.0, 1.0, 0.0],  # H
            [0.0, 0.0, 1.0],  # H
            [-1.0, 0.0, 0.0],  # H
        ])
        groups = torch.tensor([14, 1, 1, 1, 1])  # C (group 14), H (group 1)
        periods = torch.tensor([2, 1, 1, 1, 1])  # C (period 2), H (period 1)

        mol = coords_group_period_to_mol(coords, groups, periods)
        self.assertEqual(mol.GetNumAtoms(), 5)
        self.assertEqual(mol.GetNumBonds(), 4)

        # Check that the molecule is methane
        smiles = Chem.MolToSmiles(mol)
        self.assertEqual(smiles, 'C')

    def test_coords_group_period_to_mol_water(self):
        # Test with water molecule
        coords = torch.tensor([
            [0.0, 0.0, 0.0],  # O
            [0.9572, 0.0, 0.0],  # H
            [-0.2400, 0.9270, 0.0],  # H
        ])
        groups = torch.tensor([16, 1, 1])  # O (group 16), H (group 1)
        periods = torch.tensor([2, 1, 1])  # O (period 2), H (period 1)

        mol = coords_group_period_to_mol(coords, groups, periods)
        self.assertEqual(mol.GetNumAtoms(), 3)
        self.assertEqual(mol.GetNumBonds(), 2)

        # Check that the molecule is water
        smiles = Chem.MolToSmiles(mol)
        self.assertEqual(smiles, 'O')

if __name__ == '__main__':
    unittest.main()