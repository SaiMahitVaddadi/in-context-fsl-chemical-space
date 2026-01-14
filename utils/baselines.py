"""String-based baseline methods for SMILES similarity."""

from typing import List, Tuple, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter

from .smiles_utils import (
    tokenize_smiles_char, 
    tokenize_smiles_token,
    extract_all_ngrams,
    smiles_to_bow,
    build_bow_vocabulary
)
from .similarity import (
    edit_distance, 
    normalized_edit_distance, 
    jaro_winkler_similarity,
    cosine_similarity
)


class TfidfSimilarity:
    """TF-IDF based similarity for SMILES strings."""
    
    def __init__(self, char_level: bool = False, max_n: int = 3):
        """
        Initialize TF-IDF similarity calculator.
        
        Args:
            char_level: If True, use character-level tokenization
            max_n: Maximum n-gram size
        """
        self.char_level = char_level
        self.max_n = max_n
        self.vectorizer = None
        
    def _tokenize(self, smiles: str) -> List[str]:
        """Tokenize SMILES string."""
        if self.char_level:
            return tokenize_smiles_char(smiles)
        else:
            return tokenize_smiles_token(smiles)
    
    def fit(self, smiles_list: List[str]):
        """
        Fit TF-IDF vectorizer on SMILES list.
        
        Args:
            smiles_list: List of SMILES strings
        """
        # Preprocess: tokenize and join with spaces
        processed = [' '.join(self._tokenize(s)) for s in smiles_list]
        
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, self.max_n),
            analyzer='word',
            token_pattern=r'(?u)\S+'
        )
        self.vectorizer.fit(processed)
        
    def transform(self, smiles_list: List[str]) -> np.ndarray:
        """
        Transform SMILES to TF-IDF vectors.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            TF-IDF matrix
        """
        if self.vectorizer is None:
            raise ValueError("Must call fit() before transform()")
        
        processed = [' '.join(self._tokenize(s)) for s in smiles_list]
        return self.vectorizer.transform(processed).toarray()
    
    def fit_transform(self, smiles_list: List[str]) -> np.ndarray:
        """
        Fit and transform SMILES to TF-IDF vectors.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            TF-IDF matrix
        """
        processed = [' '.join(self._tokenize(s)) for s in smiles_list]
        self.vectorizer = TfidfVectorizer()
        return self.vectorizer.fit_transform(processed).toarray()
    
    def similarity(self, smiles1: str, smiles2: str) -> float:
        """
        Calculate TF-IDF cosine similarity between two SMILES.
        
        Args:
            smiles1: First SMILES string
            smiles2: Second SMILES string
            
        Returns:
            Cosine similarity (0 to 1)
        """
        if self.vectorizer is None:
            raise ValueError("Must call fit() before similarity()")
        
        vectors = self.transform([smiles1, smiles2])
        return cosine_similarity(vectors[0], vectors[1])


class BagOfWordsSimilarity:
    """Bag-of-Words based similarity for SMILES strings."""
    
    def __init__(self, char_level: bool = False, binary: bool = False):
        """
        Initialize BoW similarity calculator.
        
        Args:
            char_level: If True, use character-level tokenization
            binary: If True, use binary (presence/absence) features
        """
        self.char_level = char_level
        self.binary = binary
        self.vectorizer = None
        
    def _tokenize(self, smiles: str) -> List[str]:
        """Tokenize SMILES string."""
        if self.char_level:
            return tokenize_smiles_char(smiles)
        else:
            return tokenize_smiles_token(smiles)
    
    def fit(self, smiles_list: List[str]):
        """
        Fit BoW vectorizer on SMILES list.
        
        Args:
            smiles_list: List of SMILES strings
        """
        processed = [' '.join(self._tokenize(s)) for s in smiles_list]
        
        self.vectorizer = CountVectorizer(
            analyzer='word',
            token_pattern=r'(?u)\S+',
            binary=self.binary
        )
        self.vectorizer.fit(processed)
        
    def transform(self, smiles_list: List[str]) -> np.ndarray:
        """
        Transform SMILES to BoW vectors.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            BoW matrix
        """
        if self.vectorizer is None:
            raise ValueError("Must call fit() before transform()")
        
        processed = [' '.join(self._tokenize(s)) for s in smiles_list]
        return self.vectorizer.transform(processed).toarray()
    
    def fit_transform(self, smiles_list: List[str]) -> np.ndarray:
        """
        Fit and transform SMILES to BoW vectors.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            BoW matrix
        """
        processed = [' '.join(self._tokenize(s)) for s in smiles_list]
        self.vectorizer = CountVectorizer(binary=self.binary)
        return self.vectorizer.fit_transform(processed).toarray()
    
    def similarity(self, smiles1: str, smiles2: str) -> float:
        """
        Calculate BoW cosine similarity between two SMILES.
        
        Args:
            smiles1: First SMILES string
            smiles2: Second SMILES string
            
        Returns:
            Cosine similarity (0 to 1)
        """
        if self.vectorizer is None:
            raise ValueError("Must call fit() before similarity()")
        
        vectors = self.transform([smiles1, smiles2])
        return cosine_similarity(vectors[0], vectors[1])


class NGramSimilarity:
    """N-gram based similarity for SMILES strings."""
    
    def __init__(self, max_n: int = 5, char_level: bool = False):
        """
        Initialize N-gram similarity calculator.
        
        Args:
            max_n: Maximum n-gram size
            char_level: If True, use character-level tokenization
        """
        self.max_n = max_n
        self.char_level = char_level
        
    def extract_ngrams(self, smiles: str) -> List[str]:
        """
        Extract all n-grams from SMILES.
        
        Args:
            smiles: SMILES string
            
        Returns:
            List of n-grams
        """
        all_ngrams = extract_all_ngrams(smiles, max_n=self.max_n, 
                                       char_level=self.char_level)
        ngrams = []
        for n in all_ngrams:
            ngrams.extend(all_ngrams[n])
        return ngrams
    
    def jaccard_similarity(self, smiles1: str, smiles2: str) -> float:
        """
        Calculate Jaccard similarity based on n-grams.
        
        Args:
            smiles1: First SMILES string
            smiles2: Second SMILES string
            
        Returns:
            Jaccard similarity (0 to 1)
        """
        ngrams1 = set(self.extract_ngrams(smiles1))
        ngrams2 = set(self.extract_ngrams(smiles2))
        
        if len(ngrams1) == 0 and len(ngrams2) == 0:
            return 1.0
        
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def cosine_similarity(self, smiles1: str, smiles2: str) -> float:
        """
        Calculate cosine similarity based on n-gram frequencies.
        
        Args:
            smiles1: First SMILES string
            smiles2: Second SMILES string
            
        Returns:
            Cosine similarity (0 to 1)
        """
        ngrams1 = Counter(self.extract_ngrams(smiles1))
        ngrams2 = Counter(self.extract_ngrams(smiles2))
        
        # Get all unique n-grams
        all_ngrams = set(ngrams1.keys()) | set(ngrams2.keys())
        
        # Create vectors
        v1 = np.array([ngrams1.get(ng, 0) for ng in all_ngrams])
        v2 = np.array([ngrams2.get(ng, 0) for ng in all_ngrams])
        
        return cosine_similarity(v1, v2)
    
    def similarity(self, smiles1: str, smiles2: str) -> float:
        """
        Calculate similarity between two SMILES (defaults to cosine similarity).
        
        Args:
            smiles1: First SMILES string
            smiles2: Second SMILES string
            
        Returns:
            Similarity score (0 to 1)
        """
        return self.cosine_similarity(smiles1, smiles2)


class EditDistanceSimilarity:
    """Edit distance (Levenshtein) based similarity."""
    
    def __init__(self, normalized: bool = True):
        """
        Initialize edit distance similarity.
        
        Args:
            normalized: If True, normalize distance by max string length
        """
        self.normalized = normalized
    
    def distance(self, smiles1: str, smiles2: str) -> Union[int, float]:
        """
        Calculate edit distance between two SMILES.
        
        Args:
            smiles1: First SMILES string
            smiles2: Second SMILES string
            
        Returns:
            Edit distance (int if not normalized, float if normalized)
        """
        if self.normalized:
            return normalized_edit_distance(smiles1, smiles2)
        else:
            return edit_distance(smiles1, smiles2)
    
    def similarity(self, smiles1: str, smiles2: str) -> float:
        """
        Calculate edit distance similarity (1 - normalized_distance).
        
        Args:
            smiles1: First SMILES string
            smiles2: Second SMILES string
            
        Returns:
            Similarity (0 to 1, 1 = identical)
        """
        dist = normalized_edit_distance(smiles1, smiles2)
        return 1.0 - dist


class JaroWinklerSimilarity:
    """Jaro-Winkler similarity for SMILES strings."""
    
    def similarity(self, smiles1: str, smiles2: str) -> float:
        """
        Calculate Jaro-Winkler similarity between two SMILES.
        
        Args:
            smiles1: First SMILES string
            smiles2: Second SMILES string
            
        Returns:
            Jaro-Winkler similarity (0 to 1, 1 = identical)
        """
        return jaro_winkler_similarity(smiles1, smiles2)


def get_baseline_similarity(baseline_type: str, **kwargs):
    """
    Factory function to get baseline similarity calculator.
    
    Args:
        baseline_type: Type of baseline ('tfidf', 'bow', 'ngram', 'edit', 'jaro')
        **kwargs: Additional arguments for the baseline
        
    Returns:
        Baseline similarity calculator instance
    """
    if baseline_type == 'tfidf':
        return TfidfSimilarity(**kwargs)
    elif baseline_type == 'bow':
        return BagOfWordsSimilarity(**kwargs)
    elif baseline_type == 'ngram':
        return NGramSimilarity(**kwargs)
    elif baseline_type == 'edit':
        return EditDistanceSimilarity(**kwargs)
    elif baseline_type == 'jaro':
        return JaroWinklerSimilarity(**kwargs)
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")
