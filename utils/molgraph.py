"""
Molecular graph utilities using NetworkX.

Convert molecules to graphs and compute graph edit distance (GED).
"""

from typing import Optional, Dict, Any
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors


class MolecularGraph:
    """Convert molecules to NetworkX graphs and compute graph metrics."""
    
    @staticmethod
    def smiles_to_graph(smiles: str) -> Optional[nx.Graph]:
        """
        Convert SMILES string to NetworkX graph.
        
        Nodes represent atoms with attributes (atomic_num, is_aromatic, charge).
        Edges represent bonds with attributes (bond_type).
        
        Args:
            smiles: SMILES string representation of molecule
            
        Returns:
            NetworkX graph or None if invalid SMILES
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        G = nx.Graph()
        
        # Add nodes (atoms)
        for atom in mol.GetAtoms():
            G.add_node(
                atom.GetIdx(),
                atomic_num=atom.GetAtomicNum(),
                symbol=atom.GetSymbol(),
                is_aromatic=atom.GetIsAromatic(),
                charge=atom.GetFormalCharge(),
                hybridization=str(atom.GetHybridization())
            )
        
        # Add edges (bonds)
        for bond in mol.GetBonds():
            G.add_edge(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                bond_type=str(bond.GetBondType()),
                is_aromatic=bond.GetIsAromatic(),
                is_conjugated=bond.GetIsConjugated()
            )
        
        return G
    
    @staticmethod
    def compute_graph_edit_distance(
        smiles1: str,
        smiles2: str,
        node_match: Optional[callable] = None,
        edge_match: Optional[callable] = None,
        timeout: float = 5.0
    ) -> Optional[float]:
        """
        Compute graph edit distance (GED) between two molecules.
        
        GED measures the minimum number of graph edit operations needed
        to transform one graph into another.
        
        Args:
            smiles1: First SMILES string
            smiles2: Second SMILES string
            node_match: Optional function for matching nodes
            edge_match: Optional function for matching edges
            timeout: Maximum time in seconds for GED computation
            
        Returns:
            Graph edit distance or None if computation fails
        """
        g1 = MolecularGraph.smiles_to_graph(smiles1)
        g2 = MolecularGraph.smiles_to_graph(smiles2)
        
        if g1 is None or g2 is None:
            return None
        
        # Default node matching: compare atomic numbers
        if node_match is None:
            def node_match(n1, n2):
                return n1.get('atomic_num') == n2.get('atomic_num')
        
        # Default edge matching: compare bond types
        if edge_match is None:
            def edge_match(e1, e2):
                return e1.get('bond_type') == e2.get('bond_type')
        
        try:
            # Use optimized GED computation
            # Note: This can be slow for large molecules
            ged_list = list(nx.optimize_graph_edit_distance(
                g1, g2,
                node_match=node_match,
                edge_match=edge_match,
                timeout=timeout
            ))
            if ged_list:
                return ged_list[-1]  # Return best (last) result
            return None
        except Exception as e:
            print(f"Error computing GED: {e}")
            return None
    
    @staticmethod
    def compute_normalized_ged(smiles1: str, smiles2: str, **kwargs) -> Optional[float]:
        """
        Compute normalized graph edit distance (0 to 1).
        
        Normalized by the maximum of the two graph sizes.
        
        Args:
            smiles1: First SMILES string
            smiles2: Second SMILES string
            **kwargs: Additional arguments for GED computation
            
        Returns:
            Normalized GED (0 to 1) or None if computation fails
        """
        g1 = MolecularGraph.smiles_to_graph(smiles1)
        g2 = MolecularGraph.smiles_to_graph(smiles2)
        
        if g1 is None or g2 is None:
            return None
        
        ged = MolecularGraph.compute_graph_edit_distance(smiles1, smiles2, **kwargs)
        
        if ged is None:
            return None
        
        max_size = max(g1.number_of_nodes(), g2.number_of_nodes())
        if max_size == 0:
            return 0.0
        
        return ged / max_size
    
    @staticmethod
    def compute_graph_similarity_from_ged(smiles1: str, smiles2: str, **kwargs) -> Optional[float]:
        """
        Compute graph similarity (1 - normalized_GED).
        
        Returns a similarity score where 1.0 means identical graphs
        and 0.0 means completely different.
        
        Args:
            smiles1: First SMILES string
            smiles2: Second SMILES string
            **kwargs: Additional arguments for GED computation
            
        Returns:
            Graph similarity (0 to 1) or None if computation fails
        """
        norm_ged = MolecularGraph.compute_normalized_ged(smiles1, smiles2, **kwargs)
        
        if norm_ged is None:
            return None
        
        return 1.0 - norm_ged
    
    @staticmethod
    def get_graph_properties(smiles: str) -> Optional[Dict[str, Any]]:
        """
        Get basic graph properties of a molecule.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary of graph properties or None if invalid SMILES
        """
        G = MolecularGraph.smiles_to_graph(smiles)
        if G is None:
            return None
        
        mol = Chem.MolFromSmiles(smiles)
        
        properties = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'is_connected': nx.is_connected(G),
            'num_connected_components': nx.number_connected_components(G),
            'molecular_weight': Descriptors.MolWt(mol) if mol else None,
            'num_rings': Descriptors.RingCount(mol) if mol else None,
        }
        
        # Add degree statistics
        degrees = [d for n, d in G.degree()]
        if degrees:
            properties['avg_degree'] = sum(degrees) / len(degrees)
            properties['max_degree'] = max(degrees)
        
        return properties
