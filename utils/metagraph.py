"""Metagraph utilities for baseline edge weighting."""

from typing import List, Dict, Tuple, Optional
import numpy as np
from .baselines import get_baseline_similarity


class MetaGraph:
    """
    Meta-graph for few-shot learning with baseline edge weighting.
    
    Represents relationships between molecules using various similarity metrics.
    """
    
    def __init__(self, smiles_list: List[str], similarity_type: str = 'tfidf',
                 **similarity_kwargs):
        """
        Initialize meta-graph.
        
        Args:
            smiles_list: List of SMILES strings
            similarity_type: Type of similarity metric to use
            **similarity_kwargs: Additional arguments for similarity calculator
        """
        self.smiles_list = smiles_list
        self.similarity_type = similarity_type
        self.similarity_calculator = get_baseline_similarity(
            similarity_type, **similarity_kwargs
        )
        
        # Fit similarity calculator if needed
        if hasattr(self.similarity_calculator, 'fit'):
            self.similarity_calculator.fit(smiles_list)
        
        self.adjacency_matrix = None
        self.edge_weights = None
        
    def compute_adjacency_matrix(self, threshold: float = 0.0) -> np.ndarray:
        """
        Compute adjacency matrix based on similarity.
        
        Args:
            threshold: Minimum similarity for edge to exist
            
        Returns:
            Adjacency matrix (n x n)
        """
        n = len(self.smiles_list)
        adj_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.similarity_calculator.similarity(
                    self.smiles_list[i], 
                    self.smiles_list[j]
                )
                if sim >= threshold:
                    adj_matrix[i, j] = sim
                    adj_matrix[j, i] = sim
        
        self.adjacency_matrix = adj_matrix
        return adj_matrix
    
    def compute_edge_weights(self, support_indices: List[int], 
                           query_indices: List[int]) -> Dict[Tuple[int, int], float]:
        """
        Compute edge weights between support and query molecules.
        
        Args:
            support_indices: Indices of support molecules
            query_indices: Indices of query molecules
            
        Returns:
            Dictionary mapping (support_idx, query_idx) to similarity weight
        """
        edge_weights = {}
        
        for i in support_indices:
            for j in query_indices:
                sim = self.similarity_calculator.similarity(
                    self.smiles_list[i],
                    self.smiles_list[j]
                )
                edge_weights[(i, j)] = sim
        
        self.edge_weights = edge_weights
        return edge_weights
    
    def get_nearest_neighbors(self, query_idx: int, k: int = 5) -> List[Tuple[int, float]]:
        """
        Get k nearest neighbors for a query molecule.
        
        Args:
            query_idx: Index of query molecule
            k: Number of neighbors to return
            
        Returns:
            List of (index, similarity) tuples sorted by similarity
        """
        query_smiles = self.smiles_list[query_idx]
        similarities = []
        
        for i, smiles in enumerate(self.smiles_list):
            if i != query_idx:
                sim = self.similarity_calculator.similarity(query_smiles, smiles)
                similarities.append((i, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:k]
    
    def get_support_query_similarities(self, support_indices: List[int],
                                       query_indices: List[int]) -> np.ndarray:
        """
        Get similarity matrix between support and query sets.
        
        Args:
            support_indices: Indices of support molecules
            query_indices: Indices of query molecules
            
        Returns:
            Similarity matrix (|support| x |query|)
        """
        n_support = len(support_indices)
        n_query = len(query_indices)
        sim_matrix = np.zeros((n_support, n_query))
        
        for i, support_idx in enumerate(support_indices):
            for j, query_idx in enumerate(query_indices):
                sim = self.similarity_calculator.similarity(
                    self.smiles_list[support_idx],
                    self.smiles_list[query_idx]
                )
                sim_matrix[i, j] = sim
        
        return sim_matrix


def create_metagraph(smiles_list: List[str], similarity_type: str = 'tfidf',
                     **similarity_kwargs) -> MetaGraph:
    """
    Factory function to create a meta-graph.
    
    Args:
        smiles_list: List of SMILES strings
        similarity_type: Type of similarity metric
        **similarity_kwargs: Additional arguments for similarity calculator
        
    Returns:
        MetaGraph instance
    """
    return MetaGraph(smiles_list, similarity_type, **similarity_kwargs)
