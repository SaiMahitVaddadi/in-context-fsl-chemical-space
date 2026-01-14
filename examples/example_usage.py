#!/usr/bin/env python3
"""
Example usage of the 3D shape and color similarity framework.

This script demonstrates how to use the various modules and functions.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rdkit import Chem
import numpy as np


def example_basic_3d_similarity():
    """Example: Compute 3D similarities between two molecules."""
    print("\n=== Example 1: Basic 3D Similarity ===")
    
    from utils import generate_3d_conformer, pharmacophore_similarity, espsim_similarity
    
    # Create molecules
    smiles1 = 'CCO'
    smiles2 = 'CCCO'
    
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    # Generate 3D conformers
    mol1_3d = generate_3d_conformer(mol1)
    mol2_3d = generate_3d_conformer(mol2)
    
    # Compute similarities
    pharm_sim = pharmacophore_similarity(mol1, mol2)
    esp_sim = espsim_similarity(mol1_3d, mol2_3d)
    
    print(f"Molecules: {smiles1} vs {smiles2}")
    print(f"  Pharmacophore similarity: {pharm_sim:.4f}")
    print(f"  ESPSim similarity: {esp_sim:.4f}")


def example_batch_processing():
    """Example: Compute similarities for multiple molecules."""
    print("\n=== Example 2: Batch Processing ===")
    
    from utils import compute_shape_color_similarities
    
    # List of molecules
    smiles_list = [
        'CCO',           # Ethanol
        'CC(C)O',        # Isopropanol
        'CCCO',          # Propanol
        'c1ccccc1',      # Benzene
    ]
    
    # Compute all 3D similarities
    methods = ['pharmacophore', 'espsim']
    results = compute_shape_color_similarities(smiles_list, methods)
    
    print(f"Computed similarities for {len(smiles_list)} molecules")
    print(f"Methods: {list(results.keys())}")
    
    # Print similarity matrix for pharmacophore
    print("\nPharmacophore similarity matrix:")
    pharm_matrix = results['pharmacophore']
    for i in range(len(smiles_list)):
        row_str = ' '.join([f'{pharm_matrix[i,j]:.3f}' for j in range(len(smiles_list))])
        print(f"  {smiles_list[i]:12s}: {row_str}")


def example_fingerprint_and_3d():
    """Example: Combine 2D fingerprints with 3D similarities."""
    print("\n=== Example 3: 2D Fingerprints + 3D Similarities ===")
    
    from utils import compute_similarity_matrix, compute_shape_color_similarities
    
    smiles_list = ['CCO', 'CCCO', 'c1ccccc1O']
    
    # Compute 2D fingerprint similarity (Morgan)
    morgan_sim = compute_similarity_matrix(smiles_list, 'morgan')
    
    # Compute 3D pharmacophore similarity
    results_3d = compute_shape_color_similarities(smiles_list, ['pharmacophore'])
    pharm_sim = results_3d['pharmacophore']
    
    print("Comparison of 2D vs 3D similarities:")
    print(f"{'Pair':20s} {'Morgan':>10s} {'Pharmacophore':>15s}")
    print("-" * 47)
    
    for i in range(len(smiles_list)):
        for j in range(i+1, len(smiles_list)):
            pair = f"{smiles_list[i]} - {smiles_list[j]}"
            print(f"{pair:20s} {morgan_sim[i,j]:10.4f} {pharm_sim[i,j]:15.4f}")


def example_metagraph():
    """Example: Build and analyze a meta-graph."""
    print("\n=== Example 4: Meta-Graph Construction ===")
    
    from utils import build_metagraph, compute_metagraph_statistics
    
    smiles_list = ['CCO', 'CC(C)O', 'CCCO', 'CCCCO', 'c1ccccc1']
    
    # Build meta-graph with 2D and 3D edges
    G = build_metagraph(
        smiles_list,
        fingerprint_methods=['morgan', 'maccs'],
        shape_color_types=['pharmacophore'],
        use_3d_edges=True,
        use_ged=True,
        edge_threshold=0.3  # Only include edges with similarity >= 0.3
    )
    
    # Compute statistics
    stats = compute_metagraph_statistics(G)
    
    print(f"Meta-graph with {len(smiles_list)} molecules:")
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Edges: {stats['num_edges']}")
    print(f"  Density: {stats['density']:.4f}")
    print(f"  Average degree: {stats['avg_degree']:.2f}")
    print(f"  Connected: {stats['is_connected']}")
    print(f"  Average edge weight: {stats['avg_edge_weight']:.4f}")


def example_custom_workflow():
    """Example: Custom workflow with specific requirements."""
    print("\n=== Example 5: Custom Workflow ===")
    
    from utils import (
        generate_3d_conformer,
        pharmacophore_similarity,
        espsim_similarity,
        compute_shape_color_combined
    )
    
    # Define molecules of interest
    target_smiles = 'CCO'
    candidates_smiles = ['CCCO', 'CC(C)O', 'c1ccccc1O']
    
    # Generate 3D conformers
    target_mol = generate_3d_conformer(Chem.MolFromSmiles(target_smiles))
    candidate_mols = [generate_3d_conformer(Chem.MolFromSmiles(s)) for s in candidates_smiles]
    
    # Compute similarities against target
    results = []
    for i, candidate_mol in enumerate(candidate_mols):
        pharm = pharmacophore_similarity(target_mol, candidate_mol)
        esp = espsim_similarity(target_mol, candidate_mol)
        combined = compute_shape_color_combined(target_mol, candidate_mol)
        
        results.append({
            'smiles': candidates_smiles[i],
            'pharmacophore': pharm,
            'espsim': esp,
            'combined': combined
        })
    
    # Rank candidates by combined similarity
    results.sort(key=lambda x: x['combined'], reverse=True)
    
    print(f"Ranking candidates against target: {target_smiles}")
    print(f"{'Rank':5s} {'SMILES':12s} {'Pharmacophore':15s} {'ESPSim':10s} {'Combined':10s}")
    print("-" * 60)
    
    for rank, result in enumerate(results, 1):
        print(f"{rank:5d} {result['smiles']:12s} {result['pharmacophore']:15.4f} "
              f"{result['espsim']:10.4f} {result['combined']:10.4f}")


def main():
    """Run all examples."""
    print("="*70)
    print("3D Shape and Color Similarity Framework - Example Usage")
    print("="*70)
    
    # Suppress RDKit warnings for cleaner output
    import warnings
    warnings.filterwarnings('ignore')
    
    examples = [
        example_basic_3d_similarity,
        example_batch_processing,
        example_fingerprint_and_3d,
        example_metagraph,
        example_custom_workflow,
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\nError in {example_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70)


if __name__ == '__main__':
    main()
