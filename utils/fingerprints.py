"""
Molecular fingerprint generation using RDKit.

Supports multiple fingerprint types including ECFP4, MACCS, and more.
Extensible design allows easy addition of new fingerprint types.
"""

from typing import Optional, List
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys


class FingerprintGenerator:
    """Generate molecular fingerprints from SMILES strings."""
    
    SUPPORTED_FINGERPRINTS = {
        'ECFP4': 'Extended Connectivity Fingerprint (radius=2, 2048 bits)',
        'ECFP6': 'Extended Connectivity Fingerprint (radius=3, 2048 bits)',
        'MACCS': 'MACCS Keys (166 bits)',
        'Morgan': 'Morgan Fingerprint (radius=2, 2048 bits)',
        'RDKit': 'RDKit Topological Fingerprint (2048 bits)',
        'AtomPair': 'Atom Pair Fingerprint (2048 bits)',
        'TopologicalTorsion': 'Topological Torsion Fingerprint (2048 bits)',
    }
    
    @staticmethod
    def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
        """
        Convert SMILES string to RDKit molecule object.
        
        Args:
            smiles: SMILES string representation of molecule
            
        Returns:
            RDKit molecule object or None if invalid SMILES
        """
        mol = Chem.MolFromSmiles(smiles)
        return mol
    
    @staticmethod
    def generate_ecfp4(smiles: str, n_bits: int = 2048) -> Optional[np.ndarray]:
        """
        Generate ECFP4 (Morgan) fingerprint with radius 2.
        
        Args:
            smiles: SMILES string
            n_bits: Number of bits in fingerprint (default: 2048)
            
        Returns:
            Numpy array of fingerprint bits or None if invalid SMILES
        """
        mol = FingerprintGenerator.smiles_to_mol(smiles)
        if mol is None:
            return None
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        return np.array(fp)
    
    @staticmethod
    def generate_ecfp6(smiles: str, n_bits: int = 2048) -> Optional[np.ndarray]:
        """
        Generate ECFP6 (Morgan) fingerprint with radius 3.
        
        Args:
            smiles: SMILES string
            n_bits: Number of bits in fingerprint (default: 2048)
            
        Returns:
            Numpy array of fingerprint bits or None if invalid SMILES
        """
        mol = FingerprintGenerator.smiles_to_mol(smiles)
        if mol is None:
            return None
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=n_bits)
        return np.array(fp)
    
    @staticmethod
    def generate_maccs(smiles: str) -> Optional[np.ndarray]:
        """
        Generate MACCS keys fingerprint (166 bits).
        
        Args:
            smiles: SMILES string
            
        Returns:
            Numpy array of fingerprint bits or None if invalid SMILES
        """
        mol = FingerprintGenerator.smiles_to_mol(smiles)
        if mol is None:
            return None
        
        fp = MACCSkeys.GenMACCSKeys(mol)
        return np.array(fp)
    
    @staticmethod
    def generate_morgan(smiles: str, radius: int = 2, n_bits: int = 2048) -> Optional[np.ndarray]:
        """
        Generate Morgan fingerprint with custom radius.
        
        Args:
            smiles: SMILES string
            radius: Radius for Morgan fingerprint (default: 2)
            n_bits: Number of bits in fingerprint (default: 2048)
            
        Returns:
            Numpy array of fingerprint bits or None if invalid SMILES
        """
        mol = FingerprintGenerator.smiles_to_mol(smiles)
        if mol is None:
            return None
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        return np.array(fp)
    
    @staticmethod
    def generate_rdkit_fp(smiles: str, n_bits: int = 2048) -> Optional[np.ndarray]:
        """
        Generate RDKit topological fingerprint.
        
        Args:
            smiles: SMILES string
            n_bits: Number of bits in fingerprint (default: 2048)
            
        Returns:
            Numpy array of fingerprint bits or None if invalid SMILES
        """
        mol = FingerprintGenerator.smiles_to_mol(smiles)
        if mol is None:
            return None
        
        fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
        return np.array(fp)
    
    @staticmethod
    def generate_atom_pair(smiles: str, n_bits: int = 2048) -> Optional[np.ndarray]:
        """
        Generate Atom Pair fingerprint.
        
        Args:
            smiles: SMILES string
            n_bits: Number of bits in fingerprint (default: 2048)
            
        Returns:
            Numpy array of fingerprint bits or None if invalid SMILES
        """
        mol = FingerprintGenerator.smiles_to_mol(smiles)
        if mol is None:
            return None
        
        fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
        return np.array(fp)
    
    @staticmethod
    def generate_topological_torsion(smiles: str, n_bits: int = 2048) -> Optional[np.ndarray]:
        """
        Generate Topological Torsion fingerprint.
        
        Args:
            smiles: SMILES string
            n_bits: Number of bits in fingerprint (default: 2048)
            
        Returns:
            Numpy array of fingerprint bits or None if invalid SMILES
        """
        mol = FingerprintGenerator.smiles_to_mol(smiles)
        if mol is None:
            return None
        
        fp = AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits)
        return np.array(fp)
    
    @classmethod
    def generate(cls, smiles: str, fingerprint_type: str = 'ECFP4', **kwargs) -> Optional[np.ndarray]:
        """
        Generate fingerprint of specified type.
        
        Args:
            smiles: SMILES string
            fingerprint_type: Type of fingerprint to generate
            **kwargs: Additional parameters for fingerprint generation
            
        Returns:
            Numpy array of fingerprint bits or None if invalid SMILES
            
        Raises:
            ValueError: If fingerprint type is not supported
        """
        fingerprint_type = fingerprint_type.upper()
        
        if fingerprint_type == 'ECFP4':
            return cls.generate_ecfp4(smiles, **kwargs)
        elif fingerprint_type == 'ECFP6':
            return cls.generate_ecfp6(smiles, **kwargs)
        elif fingerprint_type == 'MACCS':
            return cls.generate_maccs(smiles)
        elif fingerprint_type == 'MORGAN':
            return cls.generate_morgan(smiles, **kwargs)
        elif fingerprint_type == 'RDKIT':
            return cls.generate_rdkit_fp(smiles, **kwargs)
        elif fingerprint_type == 'ATOMPAIR':
            return cls.generate_atom_pair(smiles, **kwargs)
        elif fingerprint_type == 'TOPOLOGICALTORSION':
            return cls.generate_topological_torsion(smiles, **kwargs)
        else:
            raise ValueError(
                f"Unsupported fingerprint type: {fingerprint_type}. "
                f"Supported types: {list(cls.SUPPORTED_FINGERPRINTS.keys())}"
            )
    
    @classmethod
    def generate_multiple(cls, smiles: str, fingerprint_types: List[str], **kwargs) -> dict:
        """
        Generate multiple fingerprint types for a single molecule.
        
        Args:
            smiles: SMILES string
            fingerprint_types: List of fingerprint types to generate
            **kwargs: Additional parameters for fingerprint generation
            
        Returns:
            Dictionary mapping fingerprint type to numpy array
        """
        results = {}
        for fp_type in fingerprint_types:
            fp = cls.generate(smiles, fp_type, **kwargs)
            if fp is not None:
                results[fp_type] = fp
        return results
