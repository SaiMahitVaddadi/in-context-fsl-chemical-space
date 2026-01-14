"""
Molecular Similarity Module

Provides fingerprint-based and 3D shape/color similarity functions.
"""

from typing import List, Dict, Optional, Callable
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from .shape_color_similarity import (
    SHAPE_COLOR_METHODS,
    compute_shape_color_similarity
)


# Fingerprint-based similarity functions
def morgan_similarity(mol1: Chem.Mol, mol2: Chem.Mol, radius: int = 2, nBits: int = 2048) -> float:
    """
    Compute Morgan fingerprint similarity (Tanimoto coefficient).
    
    Args:
        mol1: First RDKit molecule
        mol2: Second RDKit molecule
        radius: Morgan fingerprint radius (default: 2)
        nBits: Number of bits in fingerprint (default: 2048)
        
    Returns:
        Tanimoto similarity between 0 and 1
    """
    if mol1 is None or mol2 is None:
        return 0.0
    
    try:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits=nBits)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits=nBits)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception as e:
        print(f"Error computing Morgan similarity: {e}")
        return 0.0


def maccs_similarity(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    """
    Compute MACCS keys similarity (Tanimoto coefficient).
    
    Args:
        mol1: First RDKit molecule
        mol2: Second RDKit molecule
        
    Returns:
        Tanimoto similarity between 0 and 1
    """
    if mol1 is None or mol2 is None:
        return 0.0
    
    try:
        from rdkit.Chem import MACCSkeys
        fp1 = MACCSkeys.GenMACCSKeys(mol1)
        fp2 = MACCSkeys.GenMACCSKeys(mol2)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception as e:
        print(f"Error computing MACCS similarity: {e}")
        return 0.0


def rdkit_similarity(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    """
    Compute RDKit fingerprint similarity (Tanimoto coefficient).
    
    Args:
        mol1: First RDKit molecule
        mol2: Second RDKit molecule
        
    Returns:
        Tanimoto similarity between 0 and 1
    """
    if mol1 is None or mol2 is None:
        return 0.0
    
    try:
        fp1 = Chem.RDKFingerprint(mol1)
        fp2 = Chem.RDKFingerprint(mol2)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception as e:
        print(f"Error computing RDKit fingerprint similarity: {e}")
        return 0.0


def topological_torsion_similarity(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    """
    Compute topological torsion fingerprint similarity.
    
    Args:
        mol1: First RDKit molecule
        mol2: Second RDKit molecule
        
    Returns:
        Tanimoto similarity between 0 and 1
    """
    if mol1 is None or mol2 is None:
        return 0.0
    
    try:
        fp1 = AllChem.GetTopologicalTorsionFingerprintAsIntVect(mol1)
        fp2 = AllChem.GetTopologicalTorsionFingerprintAsIntVect(mol2)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception as e:
        print(f"Error computing topological torsion similarity: {e}")
        return 0.0


def atom_pair_similarity(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    """
    Compute atom pair fingerprint similarity.
    
    Args:
        mol1: First RDKit molecule
        mol2: Second RDKit molecule
        
    Returns:
        Tanimoto similarity between 0 and 1
    """
    if mol1 is None or mol2 is None:
        return 0.0
    
    try:
        fp1 = AllChem.GetAtomPairFingerprintAsIntVect(mol1)
        fp2 = AllChem.GetAtomPairFingerprintAsIntVect(mol2)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception as e:
        print(f"Error computing atom pair similarity: {e}")
        return 0.0


# Combined fingerprint and 3D methods dictionary
SIMILARITY_FUNCTIONS = {
    # 2D Fingerprint methods
    'morgan': morgan_similarity,
    'maccs': maccs_similarity,
    'rdkit': rdkit_similarity,
    'topological_torsion': topological_torsion_similarity,
    'atom_pair': atom_pair_similarity,
    # 3D Shape/Color methods
    'rmsd': lambda m1, m2: compute_shape_color_similarity(m1, m2, 'rmsd'),
    'pharmacophore': lambda m1, m2: compute_shape_color_similarity(m1, m2, 'pharmacophore'),
    'espsim': lambda m1, m2: compute_shape_color_similarity(m1, m2, 'espsim'),
    'shape_color': lambda m1, m2: compute_shape_color_similarity(m1, m2, 'shape_color'),
}


def get_similarity_function(method_name: str) -> Optional[Callable]:
    """
    Get a similarity function by name.
    
    Args:
        method_name: Name of the similarity method
        
    Returns:
        Similarity function or None if not found
    """
    return SIMILARITY_FUNCTIONS.get(method_name.lower())


def compute_similarity_matrix(smiles_list: List[str], method: str = 'morgan') -> np.ndarray:
    """
    Compute similarity matrix for a list of SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings
        method: Similarity method to use
        
    Returns:
        n x n similarity matrix
    """
    molecules = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    n = len(molecules)
    
    similarity_func = get_similarity_function(method)
    if similarity_func is None:
        raise ValueError(f"Unknown similarity method: {method}")
    
    # Handle 3D methods differently
    if method in SHAPE_COLOR_METHODS:
        from .shape_color_similarity import prepare_molecules_3d
        print(f"Preparing molecules for 3D method: {method}")
        molecules = prepare_molecules_3d(smiles_list)
    
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                sim = similarity_func(molecules[i], molecules[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
    
    return similarity_matrix


def compute_multiple_similarities(smiles_list: List[str],
                                  methods: List[str]) -> Dict[str, np.ndarray]:
    """
    Compute multiple similarity matrices for a list of SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings
        methods: List of similarity methods to compute
        
    Returns:
        Dictionary mapping method names to similarity matrices
    """
    results = {}
    
    for method in methods:
        print(f"Computing {method} similarities...")
        try:
            results[method] = compute_similarity_matrix(smiles_list, method)
        except Exception as e:
            print(f"Error computing {method} similarity: {e}")
            results[method] = None
    
    return results
