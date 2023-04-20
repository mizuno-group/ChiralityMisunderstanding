from tqdm import tqdm
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.SaltRemover import SaltRemover

ORGANIC_ATOM_SET = set([1,5,6,7,8,9,15,16,17,35,53])
GOOD_N_ATOM_MAX = 50
GOOD_N_ATOM_MIN = 3
REMOVER = Chem.SaltRemover.SaltRemover()


def preprocess_smiles(smiles_list):
    """
    Preprocess list of SMILES for training.
      -Remove salt from molecule.
      -Exclude molecule with <3 or >50 heavy atoms.
      -Exclude molecule with non-organic atoms (other than H, B, C, N, O, F, P, S, Cl, Br, I).
      -Generate canonical & random SMILES.

    Parameters
    ----------
    smiles_list: list of str
        List of SMILES.
    
    Returns
    -------
    random_smiles: list of str
        List of randomized SMILES.
    canonical_smiles: list of str
        List of canonical SMILES.
    n_valid: int
        Number of valid SMILES in smiles_list.
    included_mask: np.array of int
        List of index of molecules included in random/canonical_smiles
    """

    random_smiles = []
    canonical_smiles = []
    included_mask = []
    n_valid = 0
    for i, smiles in enumerate(tqdm(smiles_list)):
        mol = Chem.MolFromSmiles(smiles)
        if type(mol) != Chem.rdchem.Mol:
            continue
        n_valid += 1
        atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        if not (set(atoms) <= ORGANIC_ATOM_SET):
            continue
        n_atom_heavy = Descriptors.HeavyAtomCount(mol)
        if n_atom_heavy < GOOD_N_ATOM_MIN or GOOD_N_ATOM_MAX < n_atom_heavy:
            continue
        mol2 = REMOVER.StripMol(mol,dontRemoveEverything=True)
        s2 = Chem.MolToSmiles(mol2)
        if "." in s2:
            mol_frags = Chem.GetMolFrags(mol2,asMols=True)
            largest = None
            largest_size = 0
            for mol in mol_frags:
                size = mol.GetNumAtoms()
                if size > largest_size:
                    largest = mol
                    largest_size = size
            mol2 = largest
        included_mask.append(i)
        canonical_smiles.append(Chem.MolToSmiles(mol2))
        nums_atom = list(range(mol2.GetNumAtoms()))
        np.random.shuffle(nums_atom)
        mol_renumbered = Chem.RenumberAtoms(mol2, nums_atom)
        random_smiles.append(Chem.MolToSmiles(mol_renumbered, canonical=False))
    return random_smiles, canonical_smiles, n_valid, np.array(included_mask)
