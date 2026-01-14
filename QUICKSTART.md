# Quick Start Guide

This guide will help you get started with the 3D Shape and Color Similarity framework.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Examples

### 1. Run CLI with Basic Fingerprints

```bash
python scripts/episode_generator.py \
    --smiles-file examples/molecules.txt \
    --output results.json \
    --verbose
```

### 2. Add 3D Shape/Color Similarities

```bash
python scripts/episode_generator.py \
    --smiles-file examples/molecules.txt \
    --shape-color-similarities pharmacophore espsim \
    --output results_3d.json \
    --verbose
```

### 3. Build Meta-Graph with All Features

```bash
python scripts/episode_generator.py \
    --smiles-file examples/molecules.txt \
    --fingerprints morgan maccs \
    --shape-color-similarities all \
    --build-metagraph \
    --use-3d-edges \
    --output full_results.json \
    --verbose
```

### 4. Regression Task with Labels

```bash
python scripts/episode_generator.py \
    --smiles-file examples/molecules.txt \
    --labels-file examples/properties.txt \
    --task-type regression \
    --shape-color-similarities espsim \
    --output regression_results.json
```

## Running Tests

```bash
# Run comprehensive integration tests
python tests/test_integration.py

# Run example usage demonstrations
python examples/example_usage.py
```

## Using the Python API

```python
from rdkit import Chem
from utils import (
    generate_3d_conformer,
    pharmacophore_similarity,
    espsim_similarity,
    compute_shape_color_similarities,
    build_metagraph
)

# Create molecules
mol1 = Chem.MolFromSmiles('CCO')
mol2 = Chem.MolFromSmiles('CCCO')

# Generate 3D conformers
mol1_3d = generate_3d_conformer(mol1)
mol2_3d = generate_3d_conformer(mol2)

# Compute similarities
pharm_sim = pharmacophore_similarity(mol1, mol2)
esp_sim = espsim_similarity(mol1_3d, mol2_3d)

print(f"Pharmacophore similarity: {pharm_sim:.4f}")
print(f"ESPSim similarity: {esp_sim:.4f}")

# Batch processing
smiles_list = ['CCO', 'CCCO', 'c1ccccc1']
results = compute_shape_color_similarities(
    smiles_list, 
    methods=['pharmacophore', 'espsim']
)

# Build meta-graph
G = build_metagraph(
    smiles_list,
    fingerprint_methods=['morgan'],
    shape_color_types=['pharmacophore'],
    use_3d_edges=True
)
```

## Available Similarity Methods

### 2D Fingerprint Methods
- `morgan`: Morgan fingerprints (ECFP-like)
- `maccs`: MACCS keys
- `rdkit`: RDKit topological fingerprints
- `topological_torsion`: Topological torsion fingerprints
- `atom_pair`: Atom pair fingerprints

### 3D Shape/Color Methods
- `rmsd`: Root-mean-square deviation of 3D structures
- `pharmacophore`: Pharmacophore feature overlap
- `espsim`: Electrostatic potential similarity (MIT ESPSim)
- `shape_color`: Combined shape and electrostatic features

## Output Format

The CLI generates JSON files with:

```json
{
  "smiles": ["CCO", "CCCO", ...],
  "labels": [0.5, 0.7, ...],
  "task_type": "classification",
  "similarities": {
    "morgan": [[1.0, 0.85, ...], ...],
    "espsim": [[1.0, 0.72, ...], ...]
  },
  "metagraph": {
    "statistics": {
      "num_nodes": 8,
      "num_edges": 28,
      "density": 1.0
    }
  }
}
```

## Performance Tips

1. **Start with 2D methods** - They're much faster than 3D
2. **Use batch processing** - Conformers are generated once
3. **Set edge thresholds** - Reduce graph size with `--edge-threshold 0.5`
4. **Save similarities** - Use `--save-similarities` for reuse
5. **Subset before 3D** - Filter with 2D, then apply 3D to top candidates

## Troubleshooting

### RMSD Errors
RMSD requires molecules to have similar structures. It may fail for molecules with very different sizes.

### ESPSim Not Available
If ESPSim is not installed, the framework falls back to Gasteiger charge-based similarity.

### Slow 3D Computation
3D conformer generation takes ~0.1-1s per molecule. Use smaller molecule sets or batch processing.

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [examples/example_usage.py](examples/example_usage.py) for more code examples
- Run [tests/test_integration.py](tests/test_integration.py) to verify installation

## Support

For issues or questions:
- Open an issue on GitHub
- Check the documentation in README.md
- Review example scripts in `examples/`
