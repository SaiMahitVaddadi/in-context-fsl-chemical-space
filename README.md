# In-Context Few-Shot Learning in Chemical Space

A comprehensive Python framework for few-shot learning on molecular data with integrated 2D fingerprint and 3D shape/color similarity features.

## Features

### 2D Fingerprint Similarities
- **Morgan Fingerprints**: Circular fingerprints (ECFP-like)
- **MACCS Keys**: 166-bit structural keys
- **RDKit Fingerprints**: Topological fingerprints
- **Topological Torsion**: Torsion-based descriptors
- **Atom Pairs**: Atom pair fingerprints

### 3D Shape and Color Similarities
- **RMSD**: Root-mean-square deviation of 3D conformers
- **ESPSim**: Electrostatic potential similarity (MIT package)
- **Pharmacophore**: H-bond donors/acceptors, aromatics, charges
- **Shape-Color Combined**: Weighted combination of shape and electrostatic features

### Meta-Graph Construction
- Combines 2D and 3D similarity edges
- Graph edit distance (GED) computation
- Configurable edge thresholds
- Network analysis and statistics

## Installation

```bash
# Clone the repository
git clone https://github.com/SaiMahitVaddadi/in-context-fsl-chemical-space.git
cd in-context-fsl-chemical-space

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- **rdkit**: Cheminformatics toolkit for molecular operations
- **espsim**: MIT electrostatic potential similarity package
- **numpy**: Numerical computations
- **networkx**: Graph construction and analysis
- **scipy**: Scientific computing utilities

## Usage

### Basic Example

```bash
# Generate episodes with Morgan fingerprints
python scripts/episode_generator.py --smiles-file examples/molecules.txt --output results.json
```

### Multiple Fingerprint Methods

```bash
# Use multiple 2D fingerprint methods
python scripts/episode_generator.py \
    --smiles-file examples/molecules.txt \
    --fingerprints morgan maccs rdkit \
    --output multi_fingerprint_results.json
```

### 3D Shape and Color Similarities

```bash
# Compute 3D similarities (RMSD + ESPSim)
python scripts/episode_generator.py \
    --smiles-file examples/molecules.txt \
    --shape-color-similarities rmsd espsim \
    --output 3d_results.json
```

### Complete Pipeline with Meta-Graph

```bash
# Full pipeline with 2D + 3D features and meta-graph
python scripts/episode_generator.py \
    --smiles-file examples/molecules.txt \
    --fingerprints morgan maccs \
    --shape-color-similarities rmsd espsim pharmacophore \
    --use-3d-edges \
    --build-metagraph \
    --edge-threshold 0.5 \
    --verbose \
    --output full_pipeline_results.json
```

### Regression Tasks

```bash
# For regression tasks with property values
python scripts/episode_generator.py \
    --smiles-file examples/molecules.txt \
    --labels-file examples/properties.txt \
    --task-type regression \
    --shape-color-similarities espsim \
    --output regression_results.json
```

## Command Line Options

### Input/Output
- `--smiles-file`: Path to SMILES file (required)
- `--labels-file`: Path to labels file (optional)
- `--output`: Output JSON file path

### Task Configuration
- `--task-type`: `classification` or `regression`
- `--n-way`: Number of classes (classification)
- `--k-shot`: Examples per class

### Similarity Methods
- `--fingerprints`: 2D fingerprint methods
  - Choices: `morgan`, `maccs`, `rdkit`, `topological_torsion`, `atom_pair`
- `--shape-color-similarities`: 3D similarity methods
  - Choices: `rmsd`, `espsim`, `pharmacophore`, `shape_color`, `all`

### Meta-Graph Options
- `--build-metagraph`: Enable meta-graph construction
- `--use-3d-edges`: Include 3D similarity edges
- `--use-ged`: Include graph edit distance
- `--edge-threshold`: Minimum similarity for edges

### Output Options
- `--verbose`: Detailed progress information
- `--save-similarities`: Save similarity matrices as .npy files

## Module Documentation

### utils/shape_color_descriptors.py

Core 3D descriptor functions:

```python
from utils import generate_3d_conformer, rmsd_similarity, espsim_similarity

# Generate 3D conformer
mol = Chem.MolFromSmiles('CCO')
mol_3d = generate_3d_conformer(mol)

# Compute RMSD similarity
mol1_3d = generate_3d_conformer(mol1)
mol2_3d = generate_3d_conformer(mol2)
similarity = rmsd_similarity(mol1_3d, mol2_3d)

# Compute ESPSim similarity
esp_sim = espsim_similarity(mol1_3d, mol2_3d)
```

### utils/shape_color_similarity.py

Batch processing and wrapper functions:

```python
from utils import compute_shape_color_similarities

# Compute multiple 3D similarities for a set of molecules
smiles_list = ['CCO', 'CC(C)O', 'CCCO']
similarities = compute_shape_color_similarities(
    smiles_list, 
    methods=['rmsd', 'espsim', 'pharmacophore']
)
```

### utils/similarity.py

Combined 2D and 3D similarity functions:

```python
from utils import compute_similarity_matrix, SIMILARITY_FUNCTIONS

# Compute Morgan fingerprint similarity matrix
morgan_sims = compute_similarity_matrix(smiles_list, method='morgan')

# Available methods
print(SIMILARITY_FUNCTIONS.keys())
# ['morgan', 'maccs', 'rdkit', 'topological_torsion', 'atom_pair', 
#  'rmsd', 'pharmacophore', 'espsim', 'shape_color']
```

### utils/metagraph.py

Meta-graph construction with combined features:

```python
from utils import build_metagraph, compute_metagraph_statistics

# Build meta-graph with 2D and 3D edges
G = build_metagraph(
    smiles_list,
    fingerprint_methods=['morgan', 'maccs'],
    shape_color_types=['rmsd', 'espsim'],
    use_3d_edges=True,
    edge_threshold=0.5
)

# Analyze graph
stats = compute_metagraph_statistics(G)
print(f"Nodes: {stats['num_nodes']}, Edges: {stats['num_edges']}")
```

## 3D Methods Explained

### RMSD (Root-Mean-Square Deviation)
Measures geometric similarity by comparing 3D atomic positions after optimal alignment. Lower RMSD indicates more similar shapes.

### ESPSim (Electrostatic Potential Similarity)
Uses the MIT ESPSim package to compare molecular electrostatic potentials on van der Waals surfaces. Captures charge distribution and electrostatic interactions. Falls back to Gasteiger charge correlation if ESPSim is unavailable.

### Pharmacophore Similarity
Compares key pharmacologically relevant features:
- H-bond donors and acceptors
- Aromatic rings
- Charged atoms
- Rotatable bonds

Uses Tanimoto-like overlap coefficient for feature matching.

### Shape-Color Combined
Weighted combination (default 50/50) of:
- Shape: RMSD-based geometric similarity
- Color: ESPSim electrostatic similarity

## Performance Considerations

### 3D Methods Are Slower Than 2D
- **3D conformer generation**: ~0.1-1s per molecule
- **RMSD computation**: Fast once conformers exist
- **ESPSim**: Moderate computational cost
- **2D fingerprints**: Very fast (<0.01s per molecule)

### Recommendations
1. Use 2D fingerprints for initial screening
2. Apply 3D methods to filtered subsets
3. Cache 3D conformers for repeated use
4. Use batch processing for efficiency

### Batch Processing
The framework automatically batches 3D computations:
```python
# Generates all conformers once, then computes similarities
similarities = compute_shape_color_similarities(smiles_list, methods=['rmsd', 'espsim'])
```

## File Format Examples

### SMILES File (molecules.txt)
```
CCO
CC(C)O
CCCO
c1ccccc1
```

### Labels File (properties.txt)
```
0.5
0.7
0.6
0.9
```

Or for classification:
```
active
inactive
active
active
```

## Output Format

Results are saved as JSON with:
- SMILES strings
- Labels (if provided)
- Similarity matrices (all methods)
- Meta-graph statistics
- Task configuration

Example output structure:
```json
{
  "smiles": ["CCO", "CC(C)O", ...],
  "labels": [0.5, 0.7, ...],
  "task_type": "regression",
  "similarities": {
    "morgan": [[1.0, 0.85, ...], ...],
    "espsim": [[1.0, 0.72, ...], ...],
    ...
  },
  "metagraph": {
    "statistics": {
      "num_nodes": 100,
      "num_edges": 450,
      "density": 0.09,
      ...
    }
  }
}
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{in_context_fsl_chemical_space,
  title={In-Context Few-Shot Learning in Chemical Space},
  author={SaiMahitVaddadi},
  year={2026},
  url={https://github.com/SaiMahitVaddadi/in-context-fsl-chemical-space}
}
```

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## References

- ESPSim: https://github.com/hesther/espsim
- RDKit: https://www.rdkit.org/
- NetworkX: https://networkx.org/