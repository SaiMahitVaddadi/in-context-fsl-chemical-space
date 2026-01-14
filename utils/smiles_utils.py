"""SMILES tokenization and feature extraction utilities."""

import re
from typing import List, Tuple, Union
import numpy as np
from collections import Counter


def tokenize_smiles_char(smiles: str) -> List[str]:
    """
    Tokenize SMILES at character level.
    
    Args:
        smiles: SMILES string
        
    Returns:
        List of character tokens
    """
    return list(smiles)


def tokenize_smiles_token(smiles: str) -> List[str]:
    """
    Tokenize SMILES at token level (atoms, bonds, branches).
    
    Recognizes:
    - Cl, Br, and other two-letter elements
    - Single characters for common elements
    - Numbers, bonds, branches
    
    Args:
        smiles: SMILES string
        
    Returns:
        List of tokens
    """
    # Pattern to match SMILES tokens
    pattern = r'(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])'
    tokens = re.findall(pattern, smiles)
    return tokens


def extract_ngrams(tokens: List[str], n: int) -> List[str]:
    """
    Extract n-grams from token list.
    
    Args:
        tokens: List of tokens
        n: N-gram size (1 for unigrams, 2 for bigrams, etc.)
        
    Returns:
        List of n-grams as strings
    """
    if n <= 0:
        raise ValueError("n must be positive")
    
    if len(tokens) < n:
        return []
    
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = ''.join(tokens[i:i+n])
        ngrams.append(ngram)
    
    return ngrams


def extract_all_ngrams(smiles: str, max_n: int = 5, char_level: bool = False) -> dict:
    """
    Extract n-grams from 1 to max_n from SMILES string.
    
    Args:
        smiles: SMILES string
        max_n: Maximum n-gram size
        char_level: If True, use character-level tokenization
        
    Returns:
        Dictionary mapping n to list of n-grams
    """
    if char_level:
        tokens = tokenize_smiles_char(smiles)
    else:
        tokens = tokenize_smiles_token(smiles)
    
    all_ngrams = {}
    for n in range(1, max_n + 1):
        all_ngrams[n] = extract_ngrams(tokens, n)
    
    return all_ngrams


def smiles_to_ngram_vector(smiles: str, ngram_vocab: dict, max_n: int = 5) -> np.ndarray:
    """
    Convert SMILES to n-gram frequency vector.
    
    Args:
        smiles: SMILES string
        ngram_vocab: Dictionary mapping n-grams to indices
        max_n: Maximum n-gram size
        
    Returns:
        Frequency vector
    """
    vector = np.zeros(len(ngram_vocab))
    all_ngrams = extract_all_ngrams(smiles, max_n=max_n)
    
    ngram_counts = Counter()
    for n in all_ngrams:
        ngram_counts.update(all_ngrams[n])
    
    for ngram, count in ngram_counts.items():
        if ngram in ngram_vocab:
            vector[ngram_vocab[ngram]] = count
    
    return vector


def build_ngram_vocabulary(smiles_list: List[str], max_n: int = 5, 
                           min_freq: int = 1) -> dict:
    """
    Build n-gram vocabulary from list of SMILES.
    
    Args:
        smiles_list: List of SMILES strings
        max_n: Maximum n-gram size
        min_freq: Minimum frequency for n-gram to be included
        
    Returns:
        Dictionary mapping n-grams to indices
    """
    ngram_counts = Counter()
    
    for smiles in smiles_list:
        all_ngrams = extract_all_ngrams(smiles, max_n=max_n)
        for n in all_ngrams:
            ngram_counts.update(all_ngrams[n])
    
    # Filter by minimum frequency
    vocab = {}
    idx = 0
    for ngram, count in ngram_counts.items():
        if count >= min_freq:
            vocab[ngram] = idx
            idx += 1
    
    return vocab


def smiles_to_bow(smiles: str, vocab: dict = None, char_level: bool = False) -> Union[Counter, np.ndarray]:
    """
    Convert SMILES to bag-of-words representation.
    
    Args:
        smiles: SMILES string
        vocab: Optional vocabulary mapping tokens to indices
        char_level: If True, use character-level tokenization
        
    Returns:
        Counter object or numpy array if vocab is provided
    """
    if char_level:
        tokens = tokenize_smiles_char(smiles)
    else:
        tokens = tokenize_smiles_token(smiles)
    
    bow = Counter(tokens)
    
    if vocab is not None:
        vector = np.zeros(len(vocab))
        for token, count in bow.items():
            if token in vocab:
                vector[vocab[token]] = count
        return vector
    
    return bow


def build_bow_vocabulary(smiles_list: List[str], min_freq: int = 1, 
                        char_level: bool = False) -> dict:
    """
    Build bag-of-words vocabulary from list of SMILES.
    
    Args:
        smiles_list: List of SMILES strings
        min_freq: Minimum frequency for token to be included
        char_level: If True, use character-level tokenization
        
    Returns:
        Dictionary mapping tokens to indices
    """
    token_counts = Counter()
    
    for smiles in smiles_list:
        if char_level:
            tokens = tokenize_smiles_char(smiles)
        else:
            tokens = tokenize_smiles_token(smiles)
        token_counts.update(tokens)
    
    # Filter by minimum frequency
    vocab = {}
    idx = 0
    for token, count in token_counts.items():
        if count >= min_freq:
            vocab[token] = idx
            idx += 1
    
    return vocab
