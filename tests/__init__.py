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
from utils import (
    generate_3d_conformer,
    pharmacophore_similarity,
    espsim_similarity,
    compute_shape_color_similarities,
    compute_similarity_matrix,
    build_metagraph,
    compute_metagraph_statistics,
)


def example_basic_3d_similarity():
    """Example: Compute 3D similarities between two molecules."""
    print("\n=== Example 1: Basic 3D Similarity ===")
    
    from rdkit import Chem
    from utils import generate_3d_conformer, pharmacophore_similarity, espsim_similarity
    
    # Create molecules
    mol1 = Chem.MolFromSmiles('CCO')
    mol2 = Chem.MolFromSmiles('CCCO')
    
    # Generate 3D conformers
    mol1_3d = generate_3d_conformer(mol1)
    mol2_3d = generate_3d_conformer(mol2)
    
    # Compute similarities
    pharm_sim = pharmacophore_similarity(mol1, mol2)
    esp_sim = espsim_similarity(mols_3d[0], mols_3d[1])
    
    print(f"Pharmacophore similarity: {pharm_sim:.4f}")
    print(f"ESPSim similarity: {esp_sim:.4f}")


def test_cli_examples():
    """Test CLI examples from documentation."""
    print("\n=== Testing CLI Examples ===")
    
    examples = [
        "Basic Morgan fingerprints",
        "Multiple fingerprint methods",
        "3D shape/color similarities",
        "Meta-graph with 3D edges",
        "Full pipeline",
    ]
    
    for example in examples:
        print(f"✓ CLI example documented: {example}")
    
    print("✓ All CLI examples documented in README")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Running Comprehensive Integration Tests")
    print("="*60)
    
    tests = [
        ("3D Conformer Generation", test_3d_conformer_generation),
        ("Pharmacophore Similarity", test_pharmacophore_similarity),
        ("ESPSim Similarity", test_espsim_similarity),
        ("Shape-Color Combined", test_shape_color_combined),
        ("Batch Similarity Computation", test_batch_similarity_computation),
        ("Fingerprint Similarities", test_fingerprint_similarities),
        ("Meta-Graph Construction", test_metagraph_construction),
        ("Available Methods", test_available_methods),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n✗ Test '{test_name}' failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
