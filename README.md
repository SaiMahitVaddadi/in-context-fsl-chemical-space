# In-Context Few-Shot Learning in Chemical Space

A comprehensive framework for few-shot learning in molecular/chemical space with rich in-context feature enrichment. This framework enables sophisticated molecular similarity analysis, meta-graph construction, and integration with state-of-the-art chemistry language models.

## Features

### 🧪 Molecular Fingerprints
- **ECFP4/ECFP6**: Extended Connectivity Fingerprints (Morgan fingerprints)
- **MACCS Keys**: 166-bit structural keys
- **RDKit**: Topological fingerprints
- **Atom Pair**: Atom pair fingerprints
- **Topological Torsion**: Torsion-based fingerprints
- Extensible architecture for adding custom fingerprints

### 📊 Similarity Metrics
- **Tanimoto**: Jaccard index for molecular similarity
- **Dice**: Sørensen-Dice coefficient
- **Cosine**: Cosine similarity
- **Sokal-Sneath**: Alternative similarity measure
- **Kulczynski**: Kulczynski similarity
- **McConnaughey**: McConnaughey coefficient
- Support for pairwise similarity matrices

### 🕸️ Meta-Graph Construction
- Build graphs where molecules are nodes
- Edge features include:
  - Multiple fingerprint-based similarities
  - Graph Edit Distance (GED) metrics
- Rich structural context for in-context learning
- NetworkX-based implementation with export capabilities

### 📈 Graph Edit Distance (GED)
- Compute structural similarity via graph operations
- Normalized and raw GED options
- Configurable timeout for large molecules
- Convert molecules to NetworkX graphs
- Custom node/edge matching functions

### 🤖 Model Support

#### Language Models (LLMs)
- **ChemBERTa2**: `DeepChem/ChemBERTa-77M-MLM`
- **ChemBERTa3**: `ibm/chemberta-base-v3`
- **MolFormer**: `ibm/MolFormer-XL-both-10pct`
- **MolFormer-Large**: `ibm/MolFormer-Large-10pct`
- Easy integration with HuggingFace Transformers
- Batch embedding generation with custom pooling

#### Graph Fingerprint Models (GFMs)
- Stub support for:
  - ChemElon
  - MiniMol
  - GROVER
  - DGL-LifeSci models (AttentiveFP, GCN, GAT)
- Extensible for custom GNN implementations

### 🔤 Molecular Representations
- **SMILES**: Standard molecular string representation
- **SELFIES**: Self-referencing embedded strings (100% valid)
- Easy conversion between formats

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/SaiMahitVaddadi/in-context-fsl-chemical-space.git
cd in-context-fsl-chemical-space

# Install required packages
pip install -r requirements.txt
```

### Optional: GPU Support
For GPU acceleration with PyTorch:
```bash
# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### 1. List Available Options

```bash
# List supported fingerprints
python scripts/episode_generator.py --list-fingerprints

# List supported similarity metrics
python scripts/episode_generator.py --list-similarities

# List supported models
python scripts/episode_generator.py --list-models
```

### 2. Prepare Input Data

Create a JSON file with your molecules:

```json
[
  {"smiles": "CCO", "label": "active"},
  {"smiles": "CC(C)O", "label": "inactive"},
  {"smiles": "c1ccccc1", "label": "active"}
]
```

Or use CSV format:
```csv
smiles,label
CCO,active
CC(C)O,inactive
c1ccccc1,active
```

### 3. Generate Episode with Basic Features

```bash
python scripts/episode_generator.py \
  --input molecules.json \
  --output episode.json \
  --fingerprints ECFP4 \
  --similarities Tanimoto
```

### 4. Generate Episode with Advanced Features

```bash
python scripts/episode_generator.py \
  --input molecules.json \
  --output episode_advanced.json \
  --fingerprints ECFP4 --fingerprints MACCS \
  --similarities Tanimoto --similarities Dice \
  --meta-graph \
  --use-ged \
  --model chemberta3 \
  --encoding-type smiles \
  --device cuda
```

### 5. Use as Python Library

```python
from utils.fingerprints import FingerprintGenerator
from utils.similarity import SimilarityMetrics
from utils.metagraph import MetaGraph
from models.model_loader import ModelLoader

# Generate fingerprints
smiles = "CCO"
fp_ecfp4 = FingerprintGenerator.generate(smiles, 'ECFP4')
fp_maccs = FingerprintGenerator.generate(smiles, 'MACCS')

# Compute similarity
smiles2 = "CC(C)O"
fp2 = FingerprintGenerator.generate(smiles2, 'ECFP4')
similarity = SimilarityMetrics.tanimoto(fp_ecfp4, fp2)

# Build meta-graph
smiles_list = ["CCO", "CC(C)O", "c1ccccc1"]
metagraph = MetaGraph(smiles_list)
metagraph.build_complete_metagraph(
    fingerprint_types=['ECFP4', 'MACCS'],
    similarity_metrics=['Tanimoto', 'Dice'],
    use_ged=True
)

# Load model and get embeddings
loader = ModelLoader('chemberta3', device='cuda')
loader.load()
embeddings = loader.get_embeddings(smiles, pooling='mean')
```

## CLI Reference

### Episode Generator Options

| Option | Description | Example |
|--------|-------------|---------|
| `--input, -i` | Input file (JSON/CSV) | `-i molecules.json` |
| `--output, -o` | Output JSON file | `-o episode.json` |
| `--fingerprints, -f` | Fingerprint types (multiple) | `-f ECFP4 -f MACCS` |
| `--similarities, -s` | Similarity metrics (multiple) | `-s Tanimoto -s Dice` |
| `--meta-graph` | Build meta-graph | `--meta-graph` |
| `--use-ged` | Include graph edit distance | `--use-ged` |
| `--model, -m` | Model for embeddings | `-m chemberta3` |
| `--encoding-type, -e` | SMILES or SELFIES | `-e selfies` |
| `--support-size` | Number of support examples | `--support-size 5` |
| `--query-size` | Number of query examples | `--query-size 10` |
| `--device` | Device (cuda/cpu/auto) | `--device cuda` |

## Architecture

```
in-context-fsl-chemical-space/
├── utils/
│   ├── fingerprints.py      # Fingerprint generation
│   ├── similarity.py         # Similarity metrics
│   ├── selfies_converter.py # SMILES/SELFIES conversion
│   ├── molgraph.py          # Molecular graphs & GED
│   └── metagraph.py         # Meta-graph construction
├── models/
│   ├── model_loader.py      # LLM loader (ChemBERTa, MolFormer)
│   └── gfm_loader.py        # GFM loader stub
├── scripts/
│   └── episode_generator.py # CLI for episode generation
├── requirements.txt          # Dependencies
└── README.md                # This file
```

## Output Format

The episode generator produces JSON output with the following structure:

```json
{
  "molecules": [
    {
      "index": 0,
      "smiles": "CCO",
      "selfies": null,
      "label": "active",
      "fingerprints": {
        "ECFP4": [0, 1, 0, ...],
        "MACCS": [1, 1, 0, ...]
      },
      "graph_properties": {
        "num_nodes": 3,
        "num_edges": 2,
        "density": 0.67,
        "molecular_weight": 46.07
      },
      "model_embedding": [0.123, -0.456, ...]
    }
  ],
  "pairwise_similarities": {
    "ECFP4_Tanimoto": [[1.0, 0.8, ...], ...],
    "MACCS_Dice": [[1.0, 0.75, ...], ...]
  },
  "meta_graph": {
    "num_nodes": 3,
    "num_edges": 3,
    "edges": [
      {
        "source": 0,
        "target": 1,
        "features": {
          "ECFP4_Tanimoto": 0.8,
          "MACCS_Dice": 0.75,
          "GED_similarity": 0.9
        }
      }
    ],
    "statistics": {...}
  },
  "statistics": {
    "num_molecules": 3,
    "fingerprint_types": ["ECFP4", "MACCS"],
    "similarity_metrics": ["Tanimoto", "Dice"],
    "encoding_type": "smiles",
    "embedding_dim": 768
  }
}
```

## Extending the Framework

### Add Custom Fingerprint

```python
# In utils/fingerprints.py
@staticmethod
def generate_custom_fp(smiles: str, **kwargs) -> Optional[np.ndarray]:
    """Generate custom fingerprint."""
    mol = FingerprintGenerator.smiles_to_mol(smiles)
    if mol is None:
        return None
    # Your custom implementation
    return fp_array
```

### Add Custom Similarity Metric

```python
# In utils/similarity.py
@staticmethod
def custom_metric(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Calculate custom similarity."""
    # Your custom implementation
    return similarity_score
```

### Add Custom Model

```python
# In models/model_loader.py
SUPPORTED_MODELS = {
    'custom_model': 'organization/model-name',
    # ... existing models
}
```

## Performance Considerations

- **GED Computation**: Can be slow for large molecules. Use timeout parameter.
- **Batch Processing**: Use batch methods for multiple molecules.
- **GPU Acceleration**: Enable CUDA for model embeddings.
- **Memory**: Large meta-graphs may require significant memory.

## Dependencies

Core dependencies:
- `rdkit`: Molecular informatics
- `selfies`: SELFIES representation
- `networkx`: Graph operations
- `torch`: Deep learning framework
- `transformers`: Pre-trained models
- `click`: CLI interface
- `numpy`, `pandas`: Data processing

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{in_context_fsl_chemical_space,
  title = {In-Context Few-Shot Learning in Chemical Space},
  author = {Vaddadi, Sai Mahit},
  year = {2024},
  url = {https://github.com/SaiMahitVaddadi/in-context-fsl-chemical-space}
}
```

## License

See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

This framework builds on:
- RDKit for molecular informatics
- HuggingFace Transformers for pre-trained models
- NetworkX for graph algorithms
- SELFIES for robust molecular representations