"""String similarity and distance metrics."""

from typing import Union
import numpy as np
try:
    from Levenshtein import distance as levenshtein_distance
    from Levenshtein import jaro_winkler
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False


def edit_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein edit distance between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Edit distance (number of operations to transform s1 to s2)
    """
    if HAS_LEVENSHTEIN:
        return levenshtein_distance(s1, s2)
    else:
        # Fallback to pure Python implementation
        return _edit_distance_python(s1, s2)


def _edit_distance_python(s1: str, s2: str) -> int:
    """
    Pure Python implementation of Levenshtein distance.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Edit distance
    """
    if len(s1) < len(s2):
        return _edit_distance_python(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def normalized_edit_distance(s1: str, s2: str) -> float:
    """
    Calculate normalized edit distance (0 to 1).
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Normalized edit distance (0 = identical, 1 = completely different)
    """
    distance = edit_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0
    return distance / max_len


def jaro_winkler_similarity(s1: str, s2: str) -> float:
    """
    Calculate Jaro-Winkler similarity between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Jaro-Winkler similarity (0 to 1, 1 = identical)
    """
    if HAS_LEVENSHTEIN:
        return jaro_winkler(s1, s2)
    else:
        # Fallback to Jaro similarity
        return _jaro_similarity(s1, s2)


def _jaro_similarity(s1: str, s2: str) -> float:
    """
    Pure Python implementation of Jaro similarity.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Jaro similarity (0 to 1)
    """
    if s1 == s2:
        return 1.0
    
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    
    # Maximum distance for matches
    max_dist = max(len1, len2) // 2 - 1
    if max_dist < 1:
        max_dist = 1
    
    # Initialize match flags
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    
    matches = 0
    transpositions = 0
    
    # Find matches
    for i in range(len1):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len2)
        
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break
    
    if matches == 0:
        return 0.0
    
    # Find transpositions
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    
    # Calculate Jaro similarity
    jaro = (matches / len1 + matches / len2 + 
            (matches - transpositions / 2) / matches) / 3
    
    return jaro


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Cosine similarity (-1 to 1, 1 = identical direction)
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    
    return dot_product / (norm_v1 * norm_v2)


def tanimoto_coefficient(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate Tanimoto coefficient (Jaccard index for binary vectors).
    
    Args:
        v1: First binary vector
        v2: Second binary vector
        
    Returns:
        Tanimoto coefficient (0 to 1, 1 = identical)
    """
    intersection = np.sum(np.minimum(v1, v2))
    union = np.sum(np.maximum(v1, v2))
    
    if union == 0:
        return 0.0
    
    return intersection / union
