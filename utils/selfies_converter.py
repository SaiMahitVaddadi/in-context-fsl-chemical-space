"""
SMILES to SELFIES conversion utility.

SELFIES (SELF-referencIng Embedded Strings) is a robust molecular string representation
that guarantees 100% validity of all generated strings.
"""

from typing import Optional, List
import selfies as sf


class SELFIESConverter:
    """Convert between SMILES and SELFIES representations."""
    
    @staticmethod
    def smiles_to_selfies(smiles: str) -> Optional[str]:
        """
        Convert SMILES string to SELFIES string.
        
        Args:
            smiles: SMILES string representation of molecule
            
        Returns:
            SELFIES string or None if conversion fails
        """
        try:
            selfies = sf.encoder(smiles)
            return selfies
        except Exception as e:
            print(f"Error converting SMILES to SELFIES: {e}")
            return None
    
    @staticmethod
    def selfies_to_smiles(selfies: str) -> Optional[str]:
        """
        Convert SELFIES string to SMILES string.
        
        Args:
            selfies: SELFIES string representation of molecule
            
        Returns:
            SMILES string or None if conversion fails
        """
        try:
            smiles = sf.decoder(selfies)
            return smiles
        except Exception as e:
            print(f"Error converting SELFIES to SMILES: {e}")
            return None
    
    @staticmethod
    def batch_smiles_to_selfies(smiles_list: List[str]) -> List[Optional[str]]:
        """
        Convert batch of SMILES strings to SELFIES strings.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            List of SELFIES strings (None for failed conversions)
        """
        return [SELFIESConverter.smiles_to_selfies(s) for s in smiles_list]
    
    @staticmethod
    def batch_selfies_to_smiles(selfies_list: List[str]) -> List[Optional[str]]:
        """
        Convert batch of SELFIES strings to SMILES strings.
        
        Args:
            selfies_list: List of SELFIES strings
            
        Returns:
            List of SMILES strings (None for failed conversions)
        """
        return [SELFIESConverter.selfies_to_smiles(s) for s in selfies_list]
    
    @staticmethod
    def get_selfies_alphabet(smiles_list: List[str]) -> set:
        """
        Get the SELFIES alphabet (unique tokens) from a list of SMILES.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Set of unique SELFIES tokens
        """
        alphabet = set()
        for smiles in smiles_list:
            selfies = SELFIESConverter.smiles_to_selfies(smiles)
            if selfies:
                tokens = list(sf.split_selfies(selfies))
                alphabet.update(tokens)
        return alphabet
    
    @staticmethod
    def tokenize_selfies(selfies: str) -> List[str]:
        """
        Tokenize a SELFIES string into its component symbols.
        
        Args:
            selfies: SELFIES string
            
        Returns:
            List of SELFIES tokens
        """
        return list(sf.split_selfies(selfies))
