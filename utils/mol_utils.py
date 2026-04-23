"""
utils/mol_utils.py
Molecular featurization utilities shared across all model notebooks.
"""

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


# ── Morgan Fingerprints (for Random Forest) ────────────────────────────────

def smiles_to_morgan(smiles: str,
                     radius: int = 2,
                     n_bits: int = 2048) -> np.ndarray | None:
    """
    Convert a SMILES string to a Morgan fingerprint bit vector.
    Standard for RF/XGBoost QSAR models.

    Args:
        smiles:  SMILES string
        radius:  Morgan radius (2 = ECFP4, 3 = ECFP6)
        n_bits:  Fingerprint length

    Returns:
        numpy array of shape (n_bits,) or None if invalid SMILES
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp)


def batch_smiles_to_morgan(smiles_list: list,
                           radius: int = 2,
                           n_bits: int = 2048) -> tuple[np.ndarray, list[int]]:
    """
    Convert a list of SMILES to Morgan fingerprint matrix.

    Returns:
        X:       array of shape (n_valid, n_bits)
        valid_idx: indices of valid SMILES in the input list
    """
    fps = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        fp = smiles_to_morgan(smi, radius, n_bits)
        if fp is not None:
            fps.append(fp)
            valid_idx.append(i)
    return np.vstack(fps), valid_idx


# ── RDKit Descriptors (for classical ML) ──────────────────────────────────

DESCRIPTOR_LIST = [
    'MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors',
    'TPSA', 'NumRotatableBonds', 'NumAromaticRings',
    'NumHeteroatoms', 'RingCount', 'FractionCSP3',
    'HeavyAtomCount', 'NOCount', 'NHOHCount',
    'NumAliphaticRings', 'NumSaturatedRings',
]

def smiles_to_descriptors(smiles: str) -> np.ndarray | None:
    """
    Compute a vector of RDKit physicochemical descriptors.
    Complements Morgan FPs with interpretable features.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        desc_fns = {name: fn for name, fn in Descriptors.descList
                    if name in DESCRIPTOR_LIST}
        values = [desc_fns[name](mol) for name in DESCRIPTOR_LIST
                  if name in desc_fns]
        arr = np.array(values, dtype=np.float32)
        # Replace NaN/Inf with 0
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr
    except Exception:
        return None


# ── Graph Features (for GNN models) ───────────────────────────────────────

ATOM_FEATURES = {
    'atomic_num':   list(range(1, 119)),
    'degree':       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'formal_charge': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
    'num_hs':       [0, 1, 2, 3, 4],
    'is_aromatic':  [False, True],
    'in_ring':      [False, True],
}

def one_hot(value, choices: list) -> list:
    """One-hot encode a value against a list of choices. OOV → all-zeros."""
    return [int(value == c) for c in choices]

def atom_features(atom) -> np.ndarray:
    """32-dimensional atom feature vector for graph models."""
    feats = (
        one_hot(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']) +
        one_hot(atom.GetDegree(), ATOM_FEATURES['degree']) +
        one_hot(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) +
        one_hot(atom.GetHybridization(), ATOM_FEATURES['hybridization']) +
        one_hot(atom.GetTotalNumHs(), ATOM_FEATURES['num_hs']) +
        one_hot(atom.GetIsAromatic(), ATOM_FEATURES['is_aromatic']) +
        one_hot(atom.IsInRing(), ATOM_FEATURES['in_ring'])
    )
    return np.array(feats, dtype=np.float32)


BOND_FEATURES = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    'is_conjugated': [False, True],
    'is_in_ring':    [False, True],
    'stereo': [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ],
}

def bond_features(bond) -> np.ndarray:
    """10-dimensional bond feature vector for graph models."""
    feats = (
        one_hot(bond.GetBondType(), BOND_FEATURES['bond_type']) +
        one_hot(bond.GetIsConjugated(), BOND_FEATURES['is_conjugated']) +
        one_hot(bond.IsInRing(), BOND_FEATURES['is_in_ring']) +
        one_hot(bond.GetStereo(), BOND_FEATURES['stereo'])
    )
    return np.array(feats, dtype=np.float32)


def smiles_to_graph(smiles: str) -> dict | None:
    """
    Convert SMILES to graph representation for PyG / DeepChem.
    
    Returns dict with:
        x:          atom features  (n_atoms, atom_feat_dim)
        edge_index: bond indices   (2, n_bonds * 2) — bidirectional
        edge_attr:  bond features  (n_bonds * 2, bond_feat_dim)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Atom features
    atom_feats = [atom_features(a) for a in mol.GetAtoms()]
    x = np.stack(atom_feats, axis=0)

    # Bond features (add both directions)
    edge_index = [[], []]
    edge_attrs = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)

        edge_index[0] += [i, j]
        edge_index[1] += [j, i]
        edge_attrs += [bf, bf]

    if not edge_attrs:
        # Molecule with no bonds (single atom)
        return None

    return {
        'x':          x,
        'edge_index': np.array(edge_index, dtype=np.int64),
        'edge_attr':  np.stack(edge_attrs, axis=0),
        'smiles':     smiles,
    }


# ── Lipinski Filter ────────────────────────────────────────────────────────

def passes_lipinski(smiles: str) -> bool:
    """
    Checks Lipinski Rule of Five for drug-likeness.
    MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    from rdkit.Chem import Lipinski
    return (
        Descriptors.MolWt(mol)         <= 500 and
        Descriptors.MolLogP(mol)       <= 5   and
        Lipinski.NumHDonors(mol)       <= 5   and
        Lipinski.NumHAcceptors(mol)    <= 10
    )


def filter_lipinski(smiles_list: list) -> tuple[list, list]:
    """
    Filter a list of SMILES by Lipinski Ro5.
    Returns (passing, failing) lists.
    """
    passing = [s for s in smiles_list if passes_lipinski(s)]
    failing = [s for s in smiles_list if not passes_lipinski(s)]
    return passing, failing
