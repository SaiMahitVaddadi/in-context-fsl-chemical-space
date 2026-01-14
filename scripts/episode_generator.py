#!/usr/bin/env python3
"""
Episode Generator Script

Generates few-shot learning episodes with molecular similarity features,
including both 2D fingerprints and 3D shape/color descriptors.

Supports:
- Multiple fingerprint methods (Morgan, MACCS, RDKit, etc.)
- 3D shape/color similarities (RMSD, ESPSim, Pharmacophore)
- Meta-graph construction with combined edges
- Both classification and regression tasks
"""

import argparse
import sys
import json
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.similarity import compute_multiple_similarities, SIMILARITY_FUNCTIONS
from utils.shape_color_similarity import SHAPE_COLOR_METHODS, compute_shape_color_similarities
from utils.metagraph import build_metagraph, compute_metagraph_statistics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate few-shot learning episodes with molecular similarities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with Morgan fingerprints
  python episode_generator.py --smiles-file molecules.txt --output episodes.json
  
  # With multiple fingerprint methods
  python episode_generator.py --smiles-file molecules.txt --fingerprints morgan maccs rdkit
  
  # With 3D shape/color similarities
  python episode_generator.py --smiles-file molecules.txt --shape-color-similarities rmsd espsim
  
  # With meta-graph construction
  python episode_generator.py --smiles-file molecules.txt --use-3d-edges --build-metagraph
  
  # Full pipeline with all features
  python episode_generator.py --smiles-file molecules.txt \\
      --fingerprints morgan maccs \\
      --shape-color-similarities rmsd espsim pharmacophore \\
      --use-3d-edges --build-metagraph \\
      --output full_episodes.json
        """
    )
    
    # Input/Output
    parser.add_argument('--smiles-file', type=str, required=True,
                       help='Path to file containing SMILES strings (one per line)')
    parser.add_argument('--labels-file', type=str,
                       help='Path to file containing labels (optional, for supervised tasks)')
    parser.add_argument('--output', type=str, default='episodes.json',
                       help='Output file path (default: episodes.json)')
    
    # Task configuration
    parser.add_argument('--task-type', type=str, choices=['classification', 'regression'], 
                       default='classification',
                       help='Type of learning task (default: classification)')
    parser.add_argument('--n-way', type=int, default=5,
                       help='Number of classes for classification (default: 5)')
    parser.add_argument('--k-shot', type=int, default=5,
                       help='Number of examples per class (default: 5)')
    
    # Fingerprint methods
    parser.add_argument('--fingerprints', type=str, nargs='+',
                       choices=list(SIMILARITY_FUNCTIONS.keys()),
                       default=['morgan'],
                       help='Fingerprint similarity methods to use')
    
    # 3D shape/color methods
    parser.add_argument('--shape-color-similarities', type=str, nargs='+',
                       choices=list(SHAPE_COLOR_METHODS.keys()) + ['all'],
                       help='3D shape/color similarity methods (rmsd, espsim, pharmacophore, shape_color, all)')
    
    # Meta-graph options
    parser.add_argument('--build-metagraph', action='store_true',
                       help='Build meta-graph with similarity edges')
    parser.add_argument('--use-3d-edges', action='store_true',
                       help='Include 3D shape/color edges in meta-graph')
    parser.add_argument('--use-ged', action='store_true', default=True,
                       help='Include graph edit distance in meta-graph (default: True)')
    parser.add_argument('--edge-threshold', type=float, default=0.0,
                       help='Minimum similarity for edge creation (default: 0.0)')
    
    # Output options
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed progress information')
    parser.add_argument('--save-similarities', action='store_true',
                       help='Save similarity matrices to separate files')
    
    return parser.parse_args()


def load_smiles(filepath: str) -> list:
    """Load SMILES strings from file."""
    smiles_list = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                smiles_list.append(line)
    return smiles_list


def load_labels(filepath: str) -> list:
    """Load labels from file."""
    labels = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    # Try to parse as float for regression
                    labels.append(float(line))
                except ValueError:
                    # Keep as string for classification
                    labels.append(line)
    return labels


def generate_episodes(smiles_list, labels=None, args=None):
    """
    Generate few-shot learning episodes with similarity features.
    
    Args:
        smiles_list: List of SMILES strings
        labels: Optional list of labels
        args: Command line arguments
        
    Returns:
        Dictionary containing episode data
    """
    if args.verbose:
        print(f"Processing {len(smiles_list)} molecules...")
    
    # Initialize results
    results = {
        'smiles': smiles_list,
        'labels': labels,
        'task_type': args.task_type,
        'similarities': {},
        'metagraph': None,
    }
    
    # Compute fingerprint similarities
    if args.fingerprints:
        if args.verbose:
            print(f"Computing fingerprint similarities: {args.fingerprints}")
        
        fp_methods = [m for m in args.fingerprints if m not in SHAPE_COLOR_METHODS]
        if fp_methods:
            fp_sims = compute_multiple_similarities(smiles_list, fp_methods)
            results['similarities'].update(fp_sims)
    
    # Compute 3D shape/color similarities
    if args.shape_color_similarities:
        shape_color_methods = args.shape_color_similarities
        if 'all' in shape_color_methods:
            shape_color_methods = list(SHAPE_COLOR_METHODS.keys())
        
        if args.verbose:
            print(f"Computing 3D shape/color similarities: {shape_color_methods}")
        
        shape_color_sims = compute_shape_color_similarities(smiles_list, shape_color_methods)
        results['similarities'].update(shape_color_sims)
    
    # Build meta-graph if requested
    if args.build_metagraph:
        if args.verbose:
            print("Building meta-graph...")
        
        # Prepare methods for metagraph
        fp_methods = [m for m in args.fingerprints if m not in SHAPE_COLOR_METHODS]
        shape_color_types = None
        
        if args.use_3d_edges and args.shape_color_similarities:
            shape_color_types = args.shape_color_similarities
            if 'all' in shape_color_types:
                shape_color_types = list(SHAPE_COLOR_METHODS.keys())
        
        metagraph = build_metagraph(
            smiles_list,
            fingerprint_methods=fp_methods,
            shape_color_types=shape_color_types,
            use_3d_edges=args.use_3d_edges,
            use_ged=args.use_ged,
            edge_threshold=args.edge_threshold
        )
        
        # Compute and store graph statistics
        stats = compute_metagraph_statistics(metagraph)
        results['metagraph'] = {
            'statistics': stats,
            'num_nodes': metagraph.number_of_nodes(),
            'num_edges': metagraph.number_of_edges(),
        }
        
        if args.verbose:
            print(f"Meta-graph statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
    
    return results


def save_results(results, output_path, args):
    """Save results to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {
        'smiles': results['smiles'],
        'labels': results['labels'],
        'task_type': results['task_type'],
        'similarities': {},
        'metagraph': results['metagraph'],
    }
    
    # Convert similarity matrices to lists
    for method, sim_matrix in results['similarities'].items():
        if sim_matrix is not None:
            serializable_results['similarities'][method] = sim_matrix.tolist()
    
    # Save main results
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {output_path}")
    
    # Save individual similarity matrices if requested
    if args.save_similarities:
        for method, sim_matrix in results['similarities'].items():
            if sim_matrix is not None:
                sim_file = output_path.replace('.json', f'_{method}_similarities.npy')
                np.save(sim_file, sim_matrix)
                print(f"  {method} similarities saved to {sim_file}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load SMILES
    try:
        smiles_list = load_smiles(args.smiles_file)
        print(f"Loaded {len(smiles_list)} SMILES strings from {args.smiles_file}")
    except Exception as e:
        print(f"Error loading SMILES file: {e}")
        sys.exit(1)
    
    # Load labels if provided
    labels = None
    if args.labels_file:
        try:
            labels = load_labels(args.labels_file)
            print(f"Loaded {len(labels)} labels from {args.labels_file}")
            
            if len(labels) != len(smiles_list):
                print(f"Warning: Number of labels ({len(labels)}) doesn't match number of SMILES ({len(smiles_list)})")
        except Exception as e:
            print(f"Error loading labels file: {e}")
            sys.exit(1)
    
    # Generate episodes
    try:
        results = generate_episodes(smiles_list, labels, args)
    except Exception as e:
        print(f"Error generating episodes: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save results
    try:
        save_results(results, args.output, args)
    except Exception as e:
        print(f"Error saving results: {e}")
        sys.exit(1)
    
    print("Episode generation completed successfully!")


if __name__ == '__main__':
    main()
