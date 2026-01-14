"""
Meta-graph construction for molecular datasets.

Builds graphs where nodes are molecules and edges contain both
fingerprint-based similarity and graph edit distance features.
"""

from typing import List, Dict, Optional, Any
import numpy as np
import networkx as nx
from .fingerprints import FingerprintGenerator
from .similarity import SimilarityMetrics
from .molgraph import MolecularGraph


class MetaGraph:
    """Construct and manage meta-graphs of molecular datasets."""
    
    def __init__(self, smiles_list: List[str], labels: Optional[List[Any]] = None):
        """
        Initialize meta-graph with list of molecules.
        
        Args:
            smiles_list: List of SMILES strings
            labels: Optional list of labels for molecules
        """
        self.smiles_list = smiles_list
        self.labels = labels if labels is not None else [None] * len(smiles_list)
        self.graph = nx.Graph()
        self._add_nodes()
    
    def _add_nodes(self):
        """Add molecules as nodes to the meta-graph."""
        for idx, (smiles, label) in enumerate(zip(self.smiles_list, self.labels)):
            self.graph.add_node(
                idx,
                smiles=smiles,
                label=label
            )
    
    def add_fingerprint_similarities(
        self,
        fingerprint_types: List[str] = ['ECFP4'],
        similarity_metrics: List[str] = ['Tanimoto'],
        threshold: float = 0.0
    ):
        """
        Add edges based on fingerprint similarities.
        
        Args:
            fingerprint_types: List of fingerprint types to compute
            similarity_metrics: List of similarity metrics to use
            threshold: Minimum similarity to create an edge (default: 0.0)
        """
        n = len(self.smiles_list)
        
        # Generate fingerprints for all molecules
        fingerprints = {}
        for fp_type in fingerprint_types:
            fingerprints[fp_type] = []
            for smiles in self.smiles_list:
                fp = FingerprintGenerator.generate(smiles, fp_type)
                fingerprints[fp_type].append(fp)
        
        # Compute pairwise similarities and add edges
        for i in range(n):
            for j in range(i + 1, n):
                edge_data = {}
                
                # Compute similarities for each fingerprint type and metric
                for fp_type in fingerprint_types:
                    fp_i = fingerprints[fp_type][i]
                    fp_j = fingerprints[fp_type][j]
                    
                    if fp_i is not None and fp_j is not None:
                        for metric in similarity_metrics:
                            sim = SimilarityMetrics.calculate(fp_i, fp_j, metric)
                            edge_data[f'{fp_type}_{metric}'] = sim
                
                # Add edge if any similarity exceeds threshold
                if edge_data:
                    max_sim = max(edge_data.values())
                    if max_sim >= threshold:
                        self.graph.add_edge(i, j, **edge_data)
    
    def add_graph_edit_distances(
        self,
        normalized: bool = True,
        as_similarity: bool = True,
        timeout: float = 5.0,
        threshold: float = 0.0
    ):
        """
        Add graph edit distance (GED) features to edges.
        
        Args:
            normalized: Whether to normalize GED by graph size
            as_similarity: Convert GED to similarity (1 - normalized_GED)
            timeout: Maximum time per GED computation
            threshold: Minimum similarity to create/update an edge
        """
        n = len(self.smiles_list)
        
        for i in range(n):
            for j in range(i + 1, n):
                smiles_i = self.smiles_list[i]
                smiles_j = self.smiles_list[j]
                
                if as_similarity:
                    ged_value = MolecularGraph.compute_graph_similarity_from_ged(
                        smiles_i, smiles_j, timeout=timeout
                    )
                    edge_key = 'GED_similarity'
                elif normalized:
                    ged_value = MolecularGraph.compute_normalized_ged(
                        smiles_i, smiles_j, timeout=timeout
                    )
                    edge_key = 'GED_normalized'
                else:
                    ged_value = MolecularGraph.compute_graph_edit_distance(
                        smiles_i, smiles_j, timeout=timeout
                    )
                    edge_key = 'GED'
                
                if ged_value is not None:
                    # Check if edge exists
                    if self.graph.has_edge(i, j):
                        # Update existing edge
                        self.graph[i][j][edge_key] = ged_value
                    else:
                        # Create new edge if above threshold
                        if (as_similarity and ged_value >= threshold) or \
                           (not as_similarity and ged_value <= threshold):
                            self.graph.add_edge(i, j, **{edge_key: ged_value})
    
    def build_complete_metagraph(
        self,
        fingerprint_types: List[str] = ['ECFP4', 'MACCS'],
        similarity_metrics: List[str] = ['Tanimoto', 'Dice'],
        use_ged: bool = True,
        ged_timeout: float = 5.0,
        similarity_threshold: float = 0.0
    ):
        """
        Build complete meta-graph with all features.
        
        Args:
            fingerprint_types: List of fingerprint types to compute
            similarity_metrics: List of similarity metrics to use
            use_ged: Whether to include graph edit distance
            ged_timeout: Maximum time per GED computation
            similarity_threshold: Minimum similarity to create an edge
        """
        # Add fingerprint-based similarities
        self.add_fingerprint_similarities(
            fingerprint_types=fingerprint_types,
            similarity_metrics=similarity_metrics,
            threshold=similarity_threshold
        )
        
        # Add GED features if requested
        if use_ged:
            self.add_graph_edit_distances(
                normalized=True,
                as_similarity=True,
                timeout=ged_timeout,
                threshold=similarity_threshold
            )
    
    def get_edge_features(self, node_i: int, node_j: int) -> Optional[Dict[str, float]]:
        """
        Get all edge features between two nodes.
        
        Args:
            node_i: First node index
            node_j: Second node index
            
        Returns:
            Dictionary of edge features or None if no edge exists
        """
        if self.graph.has_edge(node_i, node_j):
            return dict(self.graph[node_i][node_j])
        return None
    
    def get_neighbors(self, node_idx: int, k: Optional[int] = None) -> List[int]:
        """
        Get k nearest neighbors of a node based on edge weights.
        
        Args:
            node_idx: Node index
            k: Number of neighbors to return (None = all neighbors)
            
        Returns:
            List of neighbor node indices
        """
        neighbors = list(self.graph.neighbors(node_idx))
        
        if k is None or k >= len(neighbors):
            return neighbors
        
        # Sort by average edge weight (similarity)
        def get_avg_similarity(neighbor_idx):
            edge_data = self.graph[node_idx][neighbor_idx]
            similarities = [v for edge_key, v in edge_data.items() if 'similarity' in edge_key.lower() or 'tanimoto' in edge_key.lower() or 'dice' in edge_key.lower()]
            return np.mean(similarities) if similarities else 0.0
        
        neighbors.sort(key=get_avg_similarity, reverse=True)
        return neighbors[:k]
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the meta-graph.
        
        Returns:
            Dictionary of graph statistics
        """
        stats = {
            'num_nodes': int(self.graph.number_of_nodes()),
            'num_edges': int(self.graph.number_of_edges()),
            'density': float(nx.density(self.graph)),
            'is_connected': bool(nx.is_connected(self.graph)),
            'num_connected_components': int(nx.number_connected_components(self.graph)),
        }
        
        if self.graph.number_of_edges() > 0:
            degrees = [d for n, d in self.graph.degree()]
            stats['avg_degree'] = float(np.mean(degrees))
            stats['max_degree'] = int(np.max(degrees))
            stats['min_degree'] = int(np.min(degrees))
        
        return stats
    
    def to_adjacency_matrix(self, edge_feature: Optional[str] = None) -> np.ndarray:
        """
        Convert meta-graph to adjacency matrix.
        
        Args:
            edge_feature: Specific edge feature to use as weight (None = binary)
            
        Returns:
            Adjacency matrix as numpy array
        """
        n = len(self.smiles_list)
        adj_matrix = np.zeros((n, n))
        
        for i, j in self.graph.edges():
            if edge_feature is None:
                weight = 1.0
            else:
                edge_data = self.graph[i][j]
                weight = edge_data.get(edge_feature, 0.0)
            
            adj_matrix[i, j] = weight
            adj_matrix[j, i] = weight
        
        return adj_matrix
    
    def save_graph(self, filepath: str):
        """
        Save meta-graph to file.
        
        Args:
            filepath: Path to save graph (supports .graphml, .gexf, .gml)
        """
        if filepath.endswith('.graphml'):
            nx.write_graphml(self.graph, filepath)
        elif filepath.endswith('.gexf'):
            nx.write_gexf(self.graph, filepath)
        elif filepath.endswith('.gml'):
            nx.write_gml(self.graph, filepath)
        else:
            raise ValueError("Unsupported file format. Use .graphml, .gexf, or .gml")
    
    @staticmethod
    def load_graph(filepath: str) -> nx.Graph:
        """
        Load meta-graph from file.
        
        Args:
            filepath: Path to graph file
            
        Returns:
            NetworkX graph
        """
        if filepath.endswith('.graphml'):
            return nx.read_graphml(filepath)
        elif filepath.endswith('.gexf'):
            return nx.read_gexf(filepath)
        elif filepath.endswith('.gml'):
            return nx.read_gml(filepath)
        else:
            raise ValueError("Unsupported file format. Use .graphml, .gexf, or .gml")
