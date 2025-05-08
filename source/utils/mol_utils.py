import torch
from rdkit import Chem
import mendeleev

def coords_atomicnum_to_mol(coords: torch.Tensor, atomic_num: torch.Tensor, removeHs: bool = True) -> Chem.Mol:
    """
    Convert coordinates and atomic numbers to an RDKit molecule.

    Args:
        coords: Tensor of shape (N, 3) containing 3D coordinates
        atomic_num: Tensor of shape (N,) containing atomic numbers
        removeHs: Whether to remove hydrogen atoms from the molecule

    Returns:
        RDKit molecule object
    """
    # Create empty molecule
    mol = Chem.RWMol()

    # Add atoms to molecule
    for atomic_num_val in atomic_num:
        atom = Chem.Atom(int(atomic_num_val.item()))
        mol.AddAtom(atom)

    # Create conformer and set coordinates
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        x, y, z = coords[i].tolist()
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(x, y, z))
    mol.AddConformer(conf)

    # Add bonds based on distance (simple heuristic)
    for i in range(mol.GetNumAtoms()):
        for j in range(i + 1, mol.GetNumAtoms()):
            dist = torch.norm(coords[i] - coords[j]).item()
            # Simple distance-based bonding heuristic
            if dist < 1.8:  # Typical single bond length
                mol.AddBond(i, j, Chem.BondType.SINGLE)

    # Convert to regular molecule and sanitize
    mol = mol.GetMol()
    Chem.SanitizeMol(mol)

    # Handle hydrogens
    # Print initial number of atoms before hydrogen manipulation
    print(f"Initial number of atoms: {mol.GetNumAtoms()}")

    if removeHs:
        # Remove hydrogen atoms and print new count
        mol = Chem.RemoveHs(mol)
        print(f"Number of atoms after removing hydrogens: {mol.GetNumAtoms()}")

    # Print final number of atoms for verification
    print(f"Final number of atoms: {mol.GetNumAtoms()}")
    return mol

def coords_group_period_to_mol(coords: torch.Tensor, group: torch.Tensor, period: torch.Tensor, removeHs: bool = True) -> Chem.Mol:
    """
    Convert coordinates, group, and period indices to an RDKit molecule.

    Args:
        coords: Tensor of shape (N, 3) containing 3D coordinates
        group: Tensor of shape (N,) containing group indices (1-18)
        period: Tensor of shape (N,) containing period indices (1-7)
        removeHs: Whether to remove hydrogen atoms from the molecule

    Returns:
        RDKit molecule object
    """
    # Create empty molecule
    mol = Chem.RWMol()

    # Add atoms to molecule
    for i in range(len(group)):
        group_val = int(group[i].item())
        period_val = int(period[i].item())

        # Find the atomic number based on group and period
        # This is a simplified approach and may not work for all elements
        # especially for elements with multiple possible positions in the periodic table
        atomic_num = None

        # Handle special cases first
        if group_val == 1 and period_val == 1:
            atomic_num = 1  # Hydrogen
        elif group_val == 2 and period_val == 1:
            atomic_num = 2  # Helium
        elif group_val == 1 and period_val == 2:
            atomic_num = 3  # Lithium
        elif group_val == 2 and period_val == 2:
            atomic_num = 4  # Beryllium
        elif group_val == 13 and period_val == 2:
            atomic_num = 5  # Boron
        elif group_val == 14 and period_val == 2:
            atomic_num = 6  # Carbon
        elif group_val == 15 and period_val == 2:
            atomic_num = 7  # Nitrogen
        elif group_val == 16 and period_val == 2:
            atomic_num = 8  # Oxygen
        elif group_val == 17 and period_val == 2:
            atomic_num = 9  # Fluorine
        elif group_val == 18 and period_val == 2:
            atomic_num = 10  # Neon
        else:
            # For other elements, use mendeleev to find the element
            # This is a simplified approach and may need refinement
            elements = mendeleev.get_all_elements()
            for element in elements:
                if element.group == group_val and element.period == period_val:
                    atomic_num = element.atomic_number
                    break

            # If no match found, default to carbon (most common in organic molecules)
            if atomic_num is None:
                print(f"Warning: No element found for group {group_val} and period {period_val}. Defaulting to carbon.")
                atomic_num = 6

        atom = Chem.Atom(atomic_num)
        mol.AddAtom(atom)

    # Create conformer and set coordinates
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        x, y, z = coords[i].tolist()
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(x, y, z))
    mol.AddConformer(conf)

    # Add bonds based on distance (simple heuristic)
    for i in range(mol.GetNumAtoms()):
        for j in range(i + 1, mol.GetNumAtoms()):
            dist = torch.norm(coords[i] - coords[j]).item()
            # Simple distance-based bonding heuristic
            if dist < 1.8:  # Typical single bond length
                mol.AddBond(i, j, Chem.BondType.SINGLE)

    # Convert to regular molecule and sanitize
    mol = mol.GetMol()
    Chem.SanitizeMol(mol)

    # Handle hydrogens
    # Print initial number of atoms before hydrogen manipulation
    print(f"Initial number of atoms: {mol.GetNumAtoms()}")

    if removeHs:
        # Remove hydrogen atoms and print new count
        mol = Chem.RemoveHs(mol)
        print(f"Number of atoms after removing hydrogens: {mol.GetNumAtoms()}")

    # Print final number of atoms for verification
    print(f"Final number of atoms: {mol.GetNumAtoms()}")
    return mol



def visualize_3d_mols(mols,
    drawing_style: str = 'stick',
    titles: list[str] = None,
    width:int=1500,
    height:int=400,
    grid:tuple=None,
  ):
    import py3Dmol
    from rdkit import Chem as rdChem
    if not grid: grid = (1, len(mols))
    drawing_style_options = [
        "line",   # Wire Model
        "cross",  # Cross Model
        "stick",  # Bar Model
        "sphere", # Space Filling Model
        "cartoon",# Display secondary structure in manga
    ]
    assert drawing_style in drawing_style_options, f"Invalid drawing style. Choose from {drawing_style_options}"
    if not isinstance(mols, list): mols = [mols]
    if titles is None: titles = ["" for _ in mols]  # Default empty titles if none provided
    assert len(titles) == len(mols), "Length of titles must match the number of molecules."

    p = py3Dmol.view(width=width, height=height, viewergrid=grid)
    nrows = grid[0]
    ncols = grid[1]


    for row_idx in range(nrows):
        for col_idx in range(ncols):
            mol_idx = row_idx * ncols + col_idx
            p.removeAllModels(viewer=(row_idx, col_idx))
            p.addModel(rdChem.MolToMolBlock(mols[mol_idx], confId=0), 'sdf', viewer=(row_idx, col_idx))
            p.setStyle({drawing_style: {}},  viewer=(row_idx, col_idx))
            if titles[mol_idx]: p.addLabel(titles[mol_idx],  viewer=(row_idx, col_idx)) # , {'position': {'x': 0, 'y': 1.5, 'z': 0}, 'backgroundColor': 'white', 'fontSize': 16}
    p.zoomTo()
    p.show()