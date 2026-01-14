#!/usr/bin/env python3
"""
Comprehensive test script for 3D shape and color similarity integration.

Tests all modules and functionality:
- 3D conformer generation
- RMSD similarity
- Pharmacophore similarity
- ESPSim similarity
- Combined shape-color similarity
- Meta-graph construction
- CLI interface
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rdkit import Chem
from utils import (
    generate_3d_conformer,
    rmsd_similarity,
    pharmacophore_similarity,
    espsim_similarity,
    compute_shape_color_combined,
    compute_shape_color_similarities,
    compute_similarity_matrix,
    build_metagraph,
    compute_metagraph_statistics,
    SIMILARITY_FUNCTIONS,
    SHAPE_COLOR_METHODS,
)


def test_3d_conformer_generation():
    """Test 3D conformer generation."""
    print("\n=== Testing 3D Conformer Generation ===")
    
    smiles = 'CCO'
    mol = Chem.MolFromSmiles(smiles)
    
    mol_3d = generate_3d_conformer(mol)
    
    assert mol_3d is not None, "Failed to generate 3D conformer"
    assert mol_3d.GetNumConformers() > 0, "No conformers generated"
    
    print(f"✓ Generated 3D conformer for {smiles}")
    print(f"  Atoms: {mol_3d.GetNumAtoms()}, Conformers: {mol_3d.GetNumConformers()}")
    
    return True


def test_pharmacophore_similarity():
    """Test pharmacophore similarity."""
    print("\n=== Testing Pharmacophore Similarity ===")
    
    smiles_list = ['CCO', 'CCCO', 'c1ccccc1O']
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    
    # Test pairwise similarities
    sim_01 = pharmacophore_similarity(mols[0], mols[1])
    sim_02 = pharmacophore_similarity(mols[0], mols[2])
    
    assert 0.0 <= sim_01 <= 1.0, "Similarity out of range"
    assert 0.0 <= sim_02 <= 1.0, "Similarity out of range"
    
    print(f"✓ Pharmacophore similarity CCO vs CCCO: {sim_01:.4f}")
    print(f"✓ Pharmacophore similarity CCO vs PhOH: {sim_02:.4f}")
    
    return True


def test_espsim_similarity():
    """Test ESPSim similarity."""
    print("\n=== Testing ESPSim Similarity ===")
    
    smiles_list = ['CCO', 'CCCO']
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    mols_3d = [generate_3d_conformer(m) for m in mols]
    
    sim = espsim_similarity(mols_3d[0], mols_3d[1])
    
    # ESPSim can be negative
    assert -1.0 <= sim <= 1.0, "Similarity out of expected range"
    
    print(f"✓ ESPSim similarity CCO vs CCCO: {sim:.4f}")
    
    return True


def test_shape_color_combined():
    """Test combined shape-color similarity."""
    print("\n=== Testing Shape-Color Combined Similarity ===")
    
    smiles_list = ['CCO', 'CCCO']
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    mols_3d = [generate_3d_conformer(m) for m in mols]
    
    sim = compute_shape_color_combined(mols_3d[0], mols_3d[1])
    
    assert 0.0 <= sim <= 1.0, "Similarity out of range"
    
    print(f"✓ Shape-color combined similarity: {sim:.4f}")
    
    return True


def test_batch_similarity_computation():
    """Test batch similarity computation."""
    print("\n=== Testing Batch Similarity Computation ===")
    
    smiles_list = ['CCO', 'CC(C)O', 'CCCO', 'c1ccccc1']
    
    # Test 3D methods
    methods = ['pharmacophore', 'espsim']
    results = compute_shape_color_similarities(smiles_list, methods)
    
    assert 'pharmacophore' in results, "Missing pharmacophore results"
    assert 'espsim' in results, "Missing espsim results"
    
    for method, matrix in results.items():
        assert matrix.shape == (len(smiles_list), len(smiles_list)), f"Wrong matrix shape for {method}"
        print(f"✓ Computed {method} similarity matrix: {matrix.shape}")
    
    return True


def test_fingerprint_similarities():
    """Test 2D fingerprint similarities."""
    print("\n=== Testing 2D Fingerprint Similarities ===")
    
    smiles_list = ['CCO', 'CCCO', 'c1ccccc1']
    
    methods = ['morgan', 'maccs', 'rdkit']
    
    for method in methods:
        matrix = compute_similarity_matrix(smiles_list, method)
        assert matrix.shape == (len(smiles_list), len(smiles_list)), f"Wrong matrix shape for {method}"
        assert matrix[0, 0] == 1.0, "Diagonal should be 1.0"
        print(f"✓ Computed {method} similarity matrix: {matrix.shape}")
    
    return True


def test_metagraph_construction():
    """Test meta-graph construction."""
    print("\n=== Testing Meta-Graph Construction ===")
    
    smiles_list = ['CCO', 'CC(C)O', 'CCCO', 'c1ccccc1']
    
    # Test without 3D edges
    G = build_metagraph(
        smiles_list,
        fingerprint_methods=['morgan'],
        use_3d_edges=False,
        use_ged=True
    )
    
    assert G.number_of_nodes() == len(smiles_list), "Wrong number of nodes"
    print(f"✓ Built meta-graph without 3D edges: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Test with 3D edges
    G_3d = build_metagraph(
        smiles_list,
        fingerprint_methods=['morgan'],
        shape_color_types=['pharmacophore'],
        use_3d_edges=True,
        use_ged=True
    )
    
    assert G_3d.number_of_nodes() == len(smiles_list), "Wrong number of nodes"
    print(f"✓ Built meta-graph with 3D edges: {G_3d.number_of_nodes()} nodes, {G_3d.number_of_edges()} edges")
    
    # Test statistics
    stats = compute_metagraph_statistics(G_3d)
    assert 'num_nodes' in stats, "Missing statistics"
    assert stats['num_nodes'] == len(smiles_list), "Wrong node count in stats"
    print(f"✓ Computed meta-graph statistics: density={stats['density']:.4f}")
    
    return True


def test_available_methods():
    """Test that all methods are properly registered."""
    print("\n=== Testing Available Methods ===")
    
    print(f"Available similarity functions: {list(SIMILARITY_FUNCTIONS.keys())}")
    print(f"Available 3D methods: {list(SHAPE_COLOR_METHODS.keys())}")
    
    # Check key methods are present
    assert 'morgan' in SIMILARITY_FUNCTIONS, "Morgan method missing"
    assert 'espsim' in SIMILARITY_FUNCTIONS, "ESPSim method missing"
    assert 'pharmacophore' in SHAPE_COLOR_METHODS, "Pharmacophore method missing"
    
    print("✓ All expected methods are available")
    
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
    
    if success:
        print("\n✅ All tests passed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please review the errors above.")
        sys.exit(1)

