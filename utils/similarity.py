"""
Molecular similarity metrics for comparing fingerprints.

Supports Tanimoto, Dice, and other similarity coefficients.
Extensible design allows easy addition of new metrics.
"""

from typing import Optional, List
import numpy as np


class SimilarityMetrics:
    """Calculate similarity between molecular fingerprints."""
    
    SUPPORTED_METRICS = {
        'Tanimoto': 'Tanimoto coefficient (Jaccard index)',
        'Dice': 'Dice coefficient (Sørensen-Dice index)',
        'Cosine': 'Cosine similarity',
        'Sokal': 'Sokal-Sneath similarity',
        'Kulczynski': 'Kulczynski similarity',
        'McConnaughey': 'McConnaughey similarity',
    }
    
    @staticmethod
    def tanimoto(fp1: np.ndarray, fp2: np.ndarray) -> float:
        """
        Calculate Tanimoto coefficient (Jaccard index).
        
        Tanimoto = |A ∩ B| / |A ∪ B|
        
        Args:
            fp1: First fingerprint as numpy array
            fp2: Second fingerprint as numpy array
            
        Returns:
            Tanimoto coefficient (0.0 to 1.0)
        """
        if fp1 is None or fp2 is None:
            return 0.0
        
        fp1 = np.array(fp1, dtype=bool)
        fp2 = np.array(fp2, dtype=bool)
        
        intersection = np.sum(fp1 & fp2)
        union = np.sum(fp1 | fp2)
        
        if union == 0:
            return 0.0
        
        return float(intersection) / float(union)
    
    @staticmethod
    def dice(fp1: np.ndarray, fp2: np.ndarray) -> float:
        """
        Calculate Dice coefficient (Sørensen-Dice index).
        
        Dice = 2 * |A ∩ B| / (|A| + |B|)
        
        Args:
            fp1: First fingerprint as numpy array
            fp2: Second fingerprint as numpy array
            
        Returns:
            Dice coefficient (0.0 to 1.0)
        """
        if fp1 is None or fp2 is None:
            return 0.0
        
        fp1 = np.array(fp1, dtype=bool)
        fp2 = np.array(fp2, dtype=bool)
        
        intersection = np.sum(fp1 & fp2)
        sum_bits = np.sum(fp1) + np.sum(fp2)
        
        if sum_bits == 0:
            return 0.0
        
        return 2.0 * float(intersection) / float(sum_bits)
    
    @staticmethod
    def cosine(fp1: np.ndarray, fp2: np.ndarray) -> float:
        """
        Calculate cosine similarity.
        
        Cosine = (A · B) / (||A|| * ||B||)
        
        Args:
            fp1: First fingerprint as numpy array
            fp2: Second fingerprint as numpy array
            
        Returns:
            Cosine similarity (0.0 to 1.0)
        """
        if fp1 is None or fp2 is None:
            return 0.0
        
        fp1 = np.array(fp1, dtype=float)
        fp2 = np.array(fp2, dtype=float)
        
        dot_product = np.dot(fp1, fp2)
        norm1 = np.linalg.norm(fp1)
        norm2 = np.linalg.norm(fp2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product) / (norm1 * norm2)
    
    @staticmethod
    def sokal(fp1: np.ndarray, fp2: np.ndarray) -> float:
        """
        Calculate Sokal-Sneath similarity.
        
        Sokal = |A ∩ B| / (2 * (|A| + |B|) - 3 * |A ∩ B|)
        
        Args:
            fp1: First fingerprint as numpy array
            fp2: Second fingerprint as numpy array
            
        Returns:
            Sokal-Sneath similarity (0.0 to 1.0)
        """
        if fp1 is None or fp2 is None:
            return 0.0
        
        fp1 = np.array(fp1, dtype=bool)
        fp2 = np.array(fp2, dtype=bool)
        
        intersection = np.sum(fp1 & fp2)
        sum_bits = np.sum(fp1) + np.sum(fp2)
        
        denominator = 2 * sum_bits - 3 * intersection
        
        if denominator == 0:
            return 0.0
        
        return float(intersection) / float(denominator)
    
    @staticmethod
    def kulczynski(fp1: np.ndarray, fp2: np.ndarray) -> float:
        """
        Calculate Kulczynski similarity.
        
        Kulczynski = 0.5 * (|A ∩ B| / |A| + |A ∩ B| / |B|)
        
        Args:
            fp1: First fingerprint as numpy array
            fp2: Second fingerprint as numpy array
            
        Returns:
            Kulczynski similarity (0.0 to 1.0)
        """
        if fp1 is None or fp2 is None:
            return 0.0
        
        fp1 = np.array(fp1, dtype=bool)
        fp2 = np.array(fp2, dtype=bool)
        
        intersection = np.sum(fp1 & fp2)
        sum_fp1 = np.sum(fp1)
        sum_fp2 = np.sum(fp2)
        
        if sum_fp1 == 0 or sum_fp2 == 0:
            return 0.0
        
        return 0.5 * (float(intersection) / sum_fp1 + float(intersection) / sum_fp2)
    
    @staticmethod
    def mcconnaughey(fp1: np.ndarray, fp2: np.ndarray) -> float:
        """
        Calculate McConnaughey similarity.
        
        McConnaughey = (|A ∩ B|² - |A| * |B|) / (|A| * |B|)
        
        Args:
            fp1: First fingerprint as numpy array
            fp2: Second fingerprint as numpy array
            
        Returns:
            McConnaughey similarity (-1.0 to 1.0)
        """
        if fp1 is None or fp2 is None:
            return 0.0
        
        fp1 = np.array(fp1, dtype=bool)
        fp2 = np.array(fp2, dtype=bool)
        
        intersection = np.sum(fp1 & fp2)
        sum_fp1 = np.sum(fp1)
        sum_fp2 = np.sum(fp2)
        
        product = sum_fp1 * sum_fp2
        
        if product == 0:
            return 0.0
        
        return (float(intersection) ** 2 - product) / product
    
    @classmethod
    def calculate(cls, fp1: np.ndarray, fp2: np.ndarray, metric: str = 'Tanimoto') -> float:
        """
        Calculate similarity using specified metric.
        
        Args:
            fp1: First fingerprint as numpy array
            fp2: Second fingerprint as numpy array
            metric: Similarity metric to use
            
        Returns:
            Similarity score
            
        Raises:
            ValueError: If metric is not supported
        """
        metric = metric.capitalize()
        
        if metric == 'Tanimoto':
            return cls.tanimoto(fp1, fp2)
        elif metric == 'Dice':
            return cls.dice(fp1, fp2)
        elif metric == 'Cosine':
            return cls.cosine(fp1, fp2)
        elif metric == 'Sokal':
            return cls.sokal(fp1, fp2)
        elif metric == 'Kulczynski':
            return cls.kulczynski(fp1, fp2)
        elif metric == 'Mcconnaughey':
            return cls.mcconnaughey(fp1, fp2)
        else:
            raise ValueError(
                f"Unsupported similarity metric: {metric}. "
                f"Supported metrics: {list(cls.SUPPORTED_METRICS.keys())}"
            )
    
    @classmethod
    def calculate_multiple(cls, fp1: np.ndarray, fp2: np.ndarray, metrics: List[str]) -> dict:
        """
        Calculate multiple similarity metrics.
        
        Args:
            fp1: First fingerprint as numpy array
            fp2: Second fingerprint as numpy array
            metrics: List of similarity metrics to calculate
            
        Returns:
            Dictionary mapping metric name to similarity score
        """
        results = {}
        for metric in metrics:
            results[metric] = cls.calculate(fp1, fp2, metric)
        return results
    
    @classmethod
    def pairwise_similarity(cls, fingerprints: List[np.ndarray], metric: str = 'Tanimoto') -> np.ndarray:
        """
        Calculate pairwise similarity matrix for a list of fingerprints.
        
        Args:
            fingerprints: List of fingerprints as numpy arrays
            metric: Similarity metric to use
            
        Returns:
            Symmetric similarity matrix as numpy array
        """
        n = len(fingerprints)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            similarity_matrix[i, i] = 1.0  # Self-similarity is 1
            for j in range(i + 1, n):
                sim = cls.calculate(fingerprints[i], fingerprints[j], metric)
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        
        return similarity_matrix
