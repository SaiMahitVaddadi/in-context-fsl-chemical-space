"""
Meta-Graph Construction Module

Builds meta-graphs with molecular similarity edges including 2D fingerprints and 3D shape/color features.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import networkx as nx
from rdkit import Chem

from .similarity import compute_similarity_matrix
from .shape_color_similarity import compute_shape_color_similarities


def compute_graph_edit_distance(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    """
    Compute approximate graph edit distance between two molecules.
    
    This is a simplified version that uses molecular properties as a proxy.
    
    Args:
        mol1: First RDKit molecule
        mol2: Second RDKit molecule
        
    Returns:
        Normalized GED score between 0 and 1 (lower = more similar)
    """
    if mol1 is None or mol2 is None:
        return 1.0
    
    try:
        # Use difference in atom/bond counts as proxy for GED
        num_atoms1 = mol1.GetNumAtoms()
        num_atoms2 = mol2.GetNumAtoms()
        num_bonds1 = mol1.GetNumBonds()
        num_bonds2 = mol2.GetNumBonds()
        
        # Compute normalized differences
        atom_diff = abs(num_atoms1 - num_atoms2) / max(num_atoms1, num_atoms2, 1)
        bond_diff = abs(num_bonds1 - num_bonds2) / max(num_bonds1, num_bonds2, 1)
        
        # Average difference
        ged = (atom_diff + bond_diff) / 2.0
        
        return min(1.0, ged)
        
    except Exception as e:
        print(f"Error computing GED: {e}")
        return 1.0


def compute_ged_similarity(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    """
    Convert GED to similarity score.
    
    Args:
        mol1: First RDKit molecule
        mol2: Second RDKit molecule
        
    Returns:
        Similarity score between 0 and 1
    """
    ged = compute_graph_edit_distance(mol1, mol2)
    return 1.0 - ged


def build_metagraph(smiles_list: List[str],
                   fingerprint_methods: List[str] = None,
                   shape_color_types: List[str] = None,
                   use_3d_edges: bool = False,
                   use_ged: bool = True,
                   edge_threshold: float = 0.0) -> nx.Graph:
    """
    Build a meta-graph with molecular similarity edges.
    
    Args:
        smiles_list: List of SMILES strings for nodes
        fingerprint_methods: List of fingerprint methods (default: ['morgan'])
        shape_color_types: List of 3D methods (default: None)
        use_3d_edges: Whether to include 3D shape/color edges
        use_ged: Whether to include graph edit distance edges
        edge_threshold: Minimum similarity threshold for edges (default: 0.0)
        
    Returns:
        NetworkX graph with similarity edges
    """
    if fingerprint_methods is None:
        fingerprint_methods = ['morgan']
    
    n = len(smiles_list)
    G = nx.Graph()
    
    # Add nodes with SMILES data
    for i, smiles in enumerate(smiles_list):
        G.add_node(i, smiles=smiles)
    
    # Compute 2D fingerprint similarities
    fingerprint_sims = {}
    for method in fingerprint_methods:
        print(f"Computing {method} fingerprint similarities...")
        try:
            sim_matrix = compute_similarity_matrix(smiles_list, method)
            fingerprint_sims[method] = sim_matrix
        except Exception as e:
            print(f"Error computing {method} similarities: {e}")
    
    # Compute 3D shape/color similarities if requested
    shape_color_sims = {}
    if use_3d_edges and shape_color_types:
        print("Computing 3D shape/color similarities...")
        shape_color_sims = compute_shape_color_similarities(smiles_list, shape_color_types)
    
    # Compute GED if requested
    ged_sims = None
    if use_ged:
        print("Computing graph edit distances...")
        molecules = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        ged_sims = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    ged_sims[i, j] = 1.0
                else:
                    sim = compute_ged_similarity(molecules[i], molecules[j])
                    ged_sims[i, j] = sim
                    ged_sims[j, i] = sim
    
    # Add edges with combined features
    for i in range(n):
        for j in range(i + 1, n):
            edge_features = {}
            
            # Add fingerprint similarities
            for method, sim_matrix in fingerprint_sims.items():
                edge_features[f'{method}_sim'] = float(sim_matrix[i, j])
            
            # Add GED similarity
            if ged_sims is not None:
                edge_features['ged_sim'] = float(ged_sims[i, j])
            
            # Add 3D shape/color similarities
            for method, sim_matrix in shape_color_sims.items():
                edge_features[f'{method}_sim'] = float(sim_matrix[i, j])
            
            # Compute average similarity for edge weight
            all_sims = list(edge_features.values())
            avg_sim = np.mean(all_sims) if all_sims else 0.0
            
            # Add edge if above threshold
            if avg_sim >= edge_threshold:
                G.add_edge(i, j, weight=avg_sim, **edge_features)
    
    return G


def get_edge_features(G: nx.Graph, node1: int, node2: int) -> Dict[str, float]:
    """
    Get all edge features between two nodes.
    
    Args:
        G: NetworkX graph
        node1: First node ID
        node2: Second node ID
        
    Returns:
        Dictionary of edge features
    """
    if G.has_edge(node1, node2):
        return dict(G[node1][node2])
    return {}


def get_node_neighborhood(G: nx.Graph, node: int, k: int = 1) -> List[int]:
    """
    Get k-hop neighborhood of a node.
    
    Args:
        G: NetworkX graph
        node: Node ID
        k: Number of hops (default: 1)
        
    Returns:
        List of neighbor node IDs
    """
    if k == 1:
        return list(G.neighbors(node))
    
    # For k > 1, use BFS
    neighbors = set()
    visited = {node}
    current_level = {node}
    
    for _ in range(k):
        next_level = set()
        for n in current_level:
            for neighbor in G.neighbors(n):
                if neighbor not in visited:
                    neighbors.add(neighbor)
                    next_level.add(neighbor)
                    visited.add(neighbor)
        current_level = next_level
    
    return list(neighbors)


def compute_metagraph_statistics(G: nx.Graph) -> Dict[str, float]:
    """
    Compute statistics about the meta-graph.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary of graph statistics
    """
    stats = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
    }
    
    # Compute average edge weight
    if G.number_of_edges() > 0:
        weights = [data['weight'] for _, _, data in G.edges(data=True)]
        stats['avg_edge_weight'] = np.mean(weights)
        stats['min_edge_weight'] = np.min(weights)
        stats['max_edge_weight'] = np.max(weights)
    else:
        stats['avg_edge_weight'] = 0.0
        stats['min_edge_weight'] = 0.0
        stats['max_edge_weight'] = 0.0
    
    # Check if connected
    stats['is_connected'] = nx.is_connected(G)
    stats['num_components'] = nx.number_connected_components(G)
    
    return stats
