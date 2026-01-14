"""
3D Shape and Color Similarity Module

Wrapper functions for all 3D shape/color similarity methods with batch processing support.
"""

from typing import List, Dict, Optional, Callable
import numpy as np
from rdkit import Chem

from .shape_color_descriptors import (
    generate_3d_conformer,
    rmsd_similarity,
    pharmacophore_similarity,
    espsim_similarity,
    compute_shape_color_combined
)


# Dictionary mapping method names to similarity functions
SHAPE_COLOR_METHODS = {
    'rmsd': rmsd_similarity,
    'pharmacophore': pharmacophore_similarity,
    'espsim': espsim_similarity,
    'shape_color': compute_shape_color_combined,
}


def get_shape_color_method(method_name: str) -> Optional[Callable]:
    """
    Get a shape/color similarity method by name.
    
    Args:
        method_name: Name of the method ('rmsd', 'pharmacophore', 'espsim', 'shape_color')
        
    Returns:
        Similarity function or None if not found
    """
    return SHAPE_COLOR_METHODS.get(method_name.lower())


def prepare_molecules_3d(smiles_list: List[str]) -> List[Optional[Chem.Mol]]:
    """
    Prepare a list of molecules with 3D conformers from SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        List of RDKit molecules with 3D conformers (None for failed conversions)
    """
    molecules = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol_3d = generate_3d_conformer(mol)
                molecules.append(mol_3d)
            else:
                molecules.append(None)
        except Exception as e:
            print(f"Error processing SMILES '{smiles}': {e}")
            molecules.append(None)
    
    return molecules


def compute_shape_color_similarity(mol1: Chem.Mol, mol2: Chem.Mol,
                                   method: str = 'shape_color') -> float:
    """
    Compute 3D shape/color similarity between two molecules.
    
    Args:
        mol1: First RDKit molecule
        mol2: Second RDKit molecule
        method: Similarity method ('rmsd', 'pharmacophore', 'espsim', 'shape_color')
        
    Returns:
        Similarity score between 0 and 1
    """
    if mol1 is None or mol2 is None:
        return 0.0
    
    similarity_func = get_shape_color_method(method)
    if similarity_func is None:
        print(f"Unknown method: {method}, defaulting to shape_color")
        similarity_func = compute_shape_color_combined
    
    try:
        return similarity_func(mol1, mol2)
    except Exception as e:
        print(f"Error computing {method} similarity: {e}")
        return 0.0


def compute_shape_color_similarities(smiles_list: List[str],
                                     methods: List[str] = None) -> Dict[str, np.ndarray]:
    """
    Compute 3D shape/color similarity matrices for a list of molecules.
    
    Args:
        smiles_list: List of SMILES strings
        methods: List of similarity methods to compute (default: all methods)
        
    Returns:
        Dictionary mapping method names to similarity matrices
    """
    if methods is None:
        methods = list(SHAPE_COLOR_METHODS.keys())
    
    # Prepare molecules with 3D conformers
    print(f"Preparing {len(smiles_list)} molecules with 3D conformers...")
    molecules = prepare_molecules_3d(smiles_list)
    
    n = len(molecules)
    results = {}
    
    # Compute similarity matrix for each method
    for method in methods:
        print(f"Computing {method} similarities...")
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sim = compute_shape_color_similarity(molecules[i], molecules[j], method)
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim  # Symmetric
        
        results[method] = similarity_matrix
    
    return results


def compute_pairwise_similarities(mol1: Chem.Mol, mol2: Chem.Mol,
                                  methods: List[str] = None) -> Dict[str, float]:
    """
    Compute all 3D shape/color similarities for a pair of molecules.
    
    Args:
        mol1: First RDKit molecule
        mol2: Second RDKit molecule
        methods: List of methods to compute (default: all methods)
        
    Returns:
        Dictionary mapping method names to similarity scores
    """
    if methods is None:
        methods = list(SHAPE_COLOR_METHODS.keys())
    
    results = {}
    for method in methods:
        results[method] = compute_shape_color_similarity(mol1, mol2, method)
    
    return results


def batch_compute_similarities(molecules: List[Chem.Mol],
                               method: str = 'shape_color',
                               reference_mol: Optional[Chem.Mol] = None) -> List[float]:
    """
    Compute similarities for a batch of molecules against a reference.
    
    If no reference is provided, computes all pairwise similarities.
    
    Args:
        molecules: List of RDKit molecules
        method: Similarity method to use
        reference_mol: Reference molecule (if None, computes pairwise)
        
    Returns:
        List of similarity scores
    """
    if reference_mol is not None:
        # Compute similarities against reference
        similarities = []
        for mol in molecules:
            sim = compute_shape_color_similarity(reference_mol, mol, method)
            similarities.append(sim)
        return similarities
    else:
        # Compute all pairwise similarities
        n = len(molecules)
        similarities = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = compute_shape_color_similarity(molecules[i], molecules[j], method)
                similarities.append(sim)
        return similarities
