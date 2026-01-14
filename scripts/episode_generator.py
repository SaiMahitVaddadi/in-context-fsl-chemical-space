#!/usr/bin/env python3
"""
Episode Generator CLI for In-Context Few-Shot Learning.

Command-line interface for generating few-shot learning episodes
with rich contextual features including fingerprints, similarities,
and graph-based metrics.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import click
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.fingerprints import FingerprintGenerator
from utils.similarity import SimilarityMetrics
from utils.selfies_converter import SELFIESConverter
from utils.molgraph import MolecularGraph
from utils.metagraph import MetaGraph
from models.model_loader import ModelLoader


def load_molecules_from_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Load molecules from JSON or CSV file.
    
    Expected JSON format:
    [
        {"smiles": "CCO", "label": "active", ...},
        ...
    ]
    
    Expected CSV format:
    smiles,label
    CCO,active
    ...
    
    Args:
        filepath: Path to input file
        
    Returns:
        List of molecule dictionaries
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            return json.load(f)
    
    elif filepath.suffix == '.csv':
        import pandas as pd
        df = pd.read_csv(filepath)
        return df.to_dict('records')
    
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def generate_episode_features(
    molecules: List[Dict[str, Any]],
    fingerprint_types: List[str],
    similarity_metrics: List[str],
    use_metagraph: bool,
    use_ged: bool,
    encoding_type: str,
    model_name: Optional[str] = None,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate episode features for few-shot learning.
    
    Args:
        molecules: List of molecule dictionaries
        fingerprint_types: List of fingerprint types to compute
        similarity_metrics: List of similarity metrics to use
        use_metagraph: Whether to build meta-graph
        use_ged: Whether to compute graph edit distance
        encoding_type: 'smiles' or 'selfies'
        model_name: Optional model for embeddings
        device: Device for model ('cuda' or 'cpu')
        
    Returns:
        Dictionary containing all episode features
    """
    smiles_list = [m['smiles'] for m in molecules]
    labels = [m.get('label', None) for m in molecules]
    
    episode_data = {
        'molecules': [],
        'meta_graph': None,
        'statistics': {}
    }
    
    # Convert to SELFIES if requested
    if encoding_type.lower() == 'selfies':
        selfies_list = SELFIESConverter.batch_smiles_to_selfies(smiles_list)
    else:
        selfies_list = [None] * len(smiles_list)
    
    # Generate fingerprints and compute features for each molecule
    for idx, (mol_data, smiles, selfies, label) in enumerate(
        zip(molecules, smiles_list, selfies_list, labels)
    ):
        mol_features = {
            'index': idx,
            'smiles': smiles,
            'selfies': selfies if encoding_type.lower() == 'selfies' else None,
            'label': label,
            'fingerprints': {},
            'graph_properties': None,
            'model_embedding': None
        }
        
        # Generate fingerprints
        for fp_type in fingerprint_types:
            fp = FingerprintGenerator.generate(smiles, fp_type)
            if fp is not None:
                mol_features['fingerprints'][fp_type] = fp.tolist()
        
        # Get graph properties
        graph_props = MolecularGraph.get_graph_properties(smiles)
        if graph_props:
            mol_features['graph_properties'] = graph_props
        
        # Add original data fields
        for key, value in mol_data.items():
            if key not in ['smiles', 'label']:
                mol_features[key] = value
        
        episode_data['molecules'].append(mol_features)
    
    # Build meta-graph if requested
    if use_metagraph:
        print("Building meta-graph...")
        metagraph = MetaGraph(smiles_list, labels)
        metagraph.build_complete_metagraph(
            fingerprint_types=fingerprint_types,
            similarity_metrics=similarity_metrics,
            use_ged=use_ged,
            ged_timeout=5.0,
            similarity_threshold=0.0
        )
        
        # Convert to serializable format
        metagraph_data = {
            'num_nodes': metagraph.graph.number_of_nodes(),
            'num_edges': metagraph.graph.number_of_edges(),
            'edges': [],
            'statistics': metagraph.get_graph_statistics()
        }
        
        # Add edge information
        for i, j in metagraph.graph.edges():
            edge_features = metagraph.get_edge_features(i, j)
            metagraph_data['edges'].append({
                'source': i,
                'target': j,
                'features': edge_features
            })
        
        episode_data['meta_graph'] = metagraph_data
    
    # Compute pairwise similarities
    if len(fingerprint_types) > 0 and len(similarity_metrics) > 0:
        print("Computing pairwise similarities...")
        pairwise_sims = {}
        
        for fp_type in fingerprint_types:
            fps = [m['fingerprints'].get(fp_type) for m in episode_data['molecules']]
            fps = [np.array(fp) if fp is not None else None for fp in fps]
            
            for metric in similarity_metrics:
                key = f'{fp_type}_{metric}'
                sim_matrix = SimilarityMetrics.pairwise_similarity(fps, metric)
                pairwise_sims[key] = sim_matrix.tolist()
        
        episode_data['pairwise_similarities'] = pairwise_sims
    
    # Get model embeddings if model specified
    if model_name:
        print(f"Loading model: {model_name}")
        try:
            loader = ModelLoader(model_name, device=device)
            loader.load()
            
            print("Generating embeddings...")
            embeddings = loader.batch_get_embeddings(
                smiles_list,
                pooling='mean',
                batch_size=32
            )
            
            # Add embeddings to molecule features
            for idx, emb in enumerate(embeddings):
                # Move to CPU if needed and convert to list
                if hasattr(emb, 'cpu'):
                    emb_array = emb.cpu().numpy()
                else:
                    emb_array = emb.numpy()
                episode_data['molecules'][idx]['model_embedding'] = emb_array.tolist()
            
            episode_data['statistics']['embedding_dim'] = embeddings.shape[1]
            
        except Exception as e:
            print(f"Warning: Failed to load model or generate embeddings: {e}")
    
    # Add statistics
    episode_data['statistics']['num_molecules'] = len(molecules)
    episode_data['statistics']['fingerprint_types'] = fingerprint_types
    episode_data['statistics']['similarity_metrics'] = similarity_metrics
    episode_data['statistics']['encoding_type'] = encoding_type
    
    return episode_data


@click.command()
@click.option(
    '--input',
    '-i',
    'input_file',
    required=False,
    type=click.Path(exists=True),
    help='Input file with molecules (JSON or CSV)'
)
@click.option(
    '--output',
    '-o',
    'output_file',
    required=False,
    type=click.Path(),
    help='Output JSON file for episode data'
)
@click.option(
    '--fingerprints',
    '-f',
    multiple=True,
    default=['ECFP4'],
    help='Fingerprint types to compute (can specify multiple)'
)
@click.option(
    '--similarities',
    '-s',
    multiple=True,
    default=['Tanimoto'],
    help='Similarity metrics to compute (can specify multiple)'
)
@click.option(
    '--meta-graph',
    is_flag=True,
    help='Build meta-graph with similarities and GED'
)
@click.option(
    '--use-ged',
    is_flag=True,
    help='Include graph edit distance in meta-graph edges'
)
@click.option(
    '--model',
    '-m',
    type=str,
    default=None,
    help='Model name for embeddings (chemberta2, chemberta3, molformer, etc.)'
)
@click.option(
    '--encoding-type',
    '-e',
    type=click.Choice(['smiles', 'selfies'], case_sensitive=False),
    default='smiles',
    help='Molecular encoding type'
)
@click.option(
    '--support-size',
    type=int,
    default=None,
    help='Number of support examples (for episode sampling)'
)
@click.option(
    '--query-size',
    type=int,
    default=None,
    help='Number of query examples (for episode sampling)'
)
@click.option(
    '--device',
    type=click.Choice(['cuda', 'cpu', 'auto'], case_sensitive=False),
    default='auto',
    help='Device for model inference'
)
@click.option(
    '--list-fingerprints',
    is_flag=True,
    help='List supported fingerprint types and exit'
)
@click.option(
    '--list-similarities',
    is_flag=True,
    help='List supported similarity metrics and exit'
)
@click.option(
    '--list-models',
    is_flag=True,
    help='List supported models and exit'
)
def main(
    input_file,
    output_file,
    fingerprints,
    similarities,
    meta_graph,
    use_ged,
    model,
    encoding_type,
    support_size,
    query_size,
    device,
    list_fingerprints,
    list_similarities,
    list_models
):
    """
    Episode Generator for In-Context Few-Shot Learning.
    
    Generate episodes with rich contextual features including:
    - Multiple fingerprint types (ECFP4, MACCS, etc.)
    - Similarity metrics (Tanimoto, Dice, etc.)
    - Meta-graph with fingerprint similarities and GED
    - Model embeddings from ChemBERTa, MolFormer, etc.
    - SMILES or SELFIES encoding
    
    Example usage:
    
        python episode_generator.py -i molecules.json -o episode.json \\
            --fingerprints ECFP4 --fingerprints MACCS \\
            --similarities Tanimoto --similarities Dice \\
            --meta-graph --use-ged \\
            --model chemberta3 --encoding-type smiles
    """
    # Handle list options
    if list_fingerprints:
        click.echo("Supported Fingerprint Types:")
        for name, desc in FingerprintGenerator.SUPPORTED_FINGERPRINTS.items():
            click.echo(f"  {name}: {desc}")
        return
    
    if list_similarities:
        click.echo("Supported Similarity Metrics:")
        for name, desc in SimilarityMetrics.SUPPORTED_METRICS.items():
            click.echo(f"  {name}: {desc}")
        return
    
    if list_models:
        click.echo("Supported Models:")
        for name, model_id in ModelLoader.list_supported_models().items():
            click.echo(f"  {name}: {model_id}")
        return
    
    # Validate required options for normal operation
    if not input_file or not output_file:
        raise click.UsageError("--input and --output are required (unless using --list-* options)")
    
    # Load molecules
    click.echo(f"Loading molecules from: {input_file}")
    molecules = load_molecules_from_file(input_file)
    click.echo(f"Loaded {len(molecules)} molecules")
    
    # Sample episode if support/query sizes specified
    if support_size or query_size:
        if support_size and query_size:
            total_needed = support_size + query_size
            if len(molecules) < total_needed:
                click.echo(
                    f"Warning: Not enough molecules ({len(molecules)}) "
                    f"for support ({support_size}) + query ({query_size})"
                )
            else:
                # Simple random sampling (could be stratified)
                import random
                random.shuffle(molecules)
                molecules = molecules[:total_needed]
                click.echo(f"Sampled {total_needed} molecules for episode")
    
    # Set device
    if device == 'auto':
        device = None
    
    # Generate episode features
    click.echo("Generating episode features...")
    episode_data = generate_episode_features(
        molecules=molecules,
        fingerprint_types=list(fingerprints),
        similarity_metrics=list(similarities),
        use_metagraph=meta_graph,
        use_ged=use_ged,
        encoding_type=encoding_type,
        model_name=model,
        device=device
    )
    
    # Save to output file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    click.echo(f"Saving episode data to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(episode_data, f, indent=2)
    
    click.echo("Done!")
    
    # Print summary
    click.echo("\nEpisode Summary:")
    click.echo(f"  Molecules: {episode_data['statistics']['num_molecules']}")
    click.echo(f"  Fingerprints: {', '.join(fingerprints)}")
    click.echo(f"  Similarities: {', '.join(similarities)}")
    click.echo(f"  Encoding: {encoding_type}")
    if meta_graph and episode_data['meta_graph']:
        click.echo(f"  Meta-graph edges: {episode_data['meta_graph']['num_edges']}")
    if model:
        click.echo(f"  Model: {model}")
        if 'embedding_dim' in episode_data['statistics']:
            click.echo(f"  Embedding dim: {episode_data['statistics']['embedding_dim']}")


if __name__ == '__main__':
    main()
