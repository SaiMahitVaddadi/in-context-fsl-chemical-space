"""
3D Shape and Color Descriptors Module

This module provides functions for:
- 3D conformer generation using RDKit
- RMSD computation and normalization
- Gasteiger partial charge calculation
- Pharmacophore feature extraction (H-bond donors/acceptors, aromatics, charges)
- ESPSim integration for electrostatic potential similarity
"""

from typing import Optional, Tuple, List, Dict
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D


def generate_3d_conformer(mol: Chem.Mol, max_attempts: int = 10) -> Optional[Chem.Mol]:
    """
    Generate a 3D conformer for a molecule using RDKit.
    
    Args:
        mol: RDKit molecule object
        max_attempts: Maximum number of attempts to generate conformer
        
    Returns:
        Molecule with 3D coordinates or None if failed
    """
    if mol is None:
        return None
    
    try:
        # Add hydrogens
        mol_h = Chem.AddHs(mol)
        
        # Generate 3D coordinates
        result = AllChem.EmbedMolecule(mol_h, randomSeed=42, maxAttempts=max_attempts)
        
        if result == -1:
            # Try with different method if first fails
            AllChem.EmbedMolecule(mol_h, useRandomCoords=True, randomSeed=42)
        
        # Optimize geometry with MMFF
        if mol_h.GetNumConformers() > 0:
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
            return mol_h
        
        return None
        
    except Exception as e:
        print(f"Error generating 3D conformer: {e}")
        return None


def compute_rmsd(mol1: Chem.Mol, mol2: Chem.Mol) -> Optional[float]:
    """
    Compute RMSD between two molecules with 3D coordinates.
    
    Args:
        mol1: First RDKit molecule with 3D coordinates
        mol2: Second RDKit molecule with 3D coordinates
        
    Returns:
        RMSD value or None if computation failed
    """
    if mol1 is None or mol2 is None:
        return None
    
    if mol1.GetNumConformers() == 0 or mol2.GetNumConformers() == 0:
        return None
    
    try:
        # Align molecules and compute RMSD
        rmsd = AllChem.GetBestRMS(mol1, mol2)
        return rmsd
    except Exception as e:
        print(f"Error computing RMSD: {e}")
        return None


def rmsd_similarity(mol1: Chem.Mol, mol2: Chem.Mol, max_rmsd: float = 5.0) -> float:
    """
    Compute normalized RMSD similarity between two molecules.
    
    Args:
        mol1: First RDKit molecule with 3D coordinates
        mol2: Second RDKit molecule with 3D coordinates
        max_rmsd: Maximum RMSD value for normalization (default: 5.0 Angstroms)
        
    Returns:
        Similarity score between 0 and 1 (1 = identical, 0 = very different)
    """
    rmsd = compute_rmsd(mol1, mol2)
    
    if rmsd is None:
        return 0.0
    
    # Normalize: similarity = 1 - (rmsd / max_rmsd)
    # Clamp between 0 and 1
    similarity = max(0.0, min(1.0, 1.0 - (rmsd / max_rmsd)))
    return similarity


def compute_gasteiger_charges(mol: Chem.Mol) -> Optional[List[float]]:
    """
    Compute Gasteiger partial charges for a molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        List of partial charges or None if failed
    """
    if mol is None:
        return None
    
    try:
        AllChem.ComputeGasteigerCharges(mol)
        charges = [float(atom.GetProp('_GasteigerCharge')) for atom in mol.GetAtoms()]
        return charges
    except Exception as e:
        print(f"Error computing Gasteiger charges: {e}")
        return None


def extract_pharmacophore_features(mol: Chem.Mol) -> Dict[str, int]:
    """
    Extract pharmacophore features from a molecule.
    
    Features include:
    - H-bond donors
    - H-bond acceptors
    - Aromatic rings
    - Positive charges
    - Negative charges
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Dictionary of feature counts
    """
    if mol is None:
        return {}
    
    try:
        features = {
            'hbd': rdMolDescriptors.CalcNumHBD(mol),  # H-bond donors
            'hba': rdMolDescriptors.CalcNumHBA(mol),  # H-bond acceptors
            'aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'aliphatic_rings': rdMolDescriptors.CalcNumAliphaticRings(mol),
            'rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
        }
        
        # Count charged atoms
        charges = compute_gasteiger_charges(mol)
        if charges:
            features['positive_charges'] = sum(1 for c in charges if c > 0.3)
            features['negative_charges'] = sum(1 for c in charges if c < -0.3)
        else:
            features['positive_charges'] = 0
            features['negative_charges'] = 0
        
        return features
        
    except Exception as e:
        print(f"Error extracting pharmacophore features: {e}")
        return {}


def pharmacophore_similarity(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    """
    Compute pharmacophore similarity between two molecules.
    
    Uses Tanimoto-like coefficient for feature overlap.
    
    Args:
        mol1: First RDKit molecule
        mol2: Second RDKit molecule
        
    Returns:
        Similarity score between 0 and 1
    """
    features1 = extract_pharmacophore_features(mol1)
    features2 = extract_pharmacophore_features(mol2)
    
    if not features1 or not features2:
        return 0.0
    
    # Compute similarity for each feature
    similarities = []
    for key in features1.keys():
        if key in features2:
            val1, val2 = features1[key], features2[key]
            # Handle zero case
            if val1 == 0 and val2 == 0:
                similarities.append(1.0)
            else:
                # Tanimoto-like: min / max
                similarities.append(min(val1, val2) / max(val1, val2) if max(val1, val2) > 0 else 0.0)
    
    # Average similarity across features
    return np.mean(similarities) if similarities else 0.0


def espsim_similarity(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    """
    Compute electrostatic potential similarity using ESPSim package.
    
    Falls back to charge-based similarity if ESPSim is unavailable or fails.
    
    Args:
        mol1: First RDKit molecule with 3D coordinates
        mol2: Second RDKit molecule with 3D coordinates
        
    Returns:
        Similarity score between 0 and 1
    """
    try:
        import espsim
        
        # Ensure molecules have 3D conformers
        if mol1.GetNumConformers() == 0:
            mol1 = generate_3d_conformer(mol1)
        if mol2.GetNumConformers() == 0:
            mol2 = generate_3d_conformer(mol2)
        
        if mol1 is None or mol2 is None:
            return _charge_based_similarity(mol1, mol2)
        
        # Compute ESPSim similarity
        similarity = espsim.GetEspSim(mol1, mol2)
        return float(similarity)
        
    except ImportError:
        print("ESPSim not available, falling back to charge-based similarity")
        return _charge_based_similarity(mol1, mol2)
    except Exception as e:
        print(f"Error computing ESPSim: {e}, falling back to charge-based similarity")
        return _charge_based_similarity(mol1, mol2)


def _charge_based_similarity(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    """
    Fallback charge-based similarity when ESPSim is unavailable.
    
    Args:
        mol1: First RDKit molecule
        mol2: Second RDKit molecule
        
    Returns:
        Similarity score between 0 and 1
    """
    if mol1 is None or mol2 is None:
        return 0.0
    
    charges1 = compute_gasteiger_charges(mol1)
    charges2 = compute_gasteiger_charges(mol2)
    
    if not charges1 or not charges2:
        return 0.0
    
    # Pad to same length
    max_len = max(len(charges1), len(charges2))
    charges1 = charges1 + [0.0] * (max_len - len(charges1))
    charges2 = charges2 + [0.0] * (max_len - len(charges2))
    
    # Compute correlation coefficient
    try:
        correlation = np.corrcoef(charges1, charges2)[0, 1]
        # Convert to similarity (0 to 1 range)
        similarity = (correlation + 1.0) / 2.0
        return max(0.0, min(1.0, similarity))
    except:
        return 0.0


def compute_shape_color_combined(mol1: Chem.Mol, mol2: Chem.Mol,
                                  shape_weight: float = 0.5,
                                  color_weight: float = 0.5) -> float:
    """
    Compute combined shape and color similarity.
    
    Shape: RMSD-based geometric similarity
    Color: ESPSim electrostatic similarity
    
    Args:
        mol1: First RDKit molecule with 3D coordinates
        mol2: Second RDKit molecule with 3D coordinates
        shape_weight: Weight for shape component (default: 0.5)
        color_weight: Weight for color component (default: 0.5)
        
    Returns:
        Combined similarity score between 0 and 1
    """
    shape_sim = rmsd_similarity(mol1, mol2)
    color_sim = espsim_similarity(mol1, mol2)
    
    combined = shape_weight * shape_sim + color_weight * color_sim
    return combined
