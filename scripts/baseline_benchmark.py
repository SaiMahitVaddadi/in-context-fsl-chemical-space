"""Standalone CLI for baseline benchmarking."""

import argparse
import time
import json
from typing import List, Dict, Any
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to allow imports when running as script
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).parent.parent))

from models.baseline_models import NGramMLBaseline, get_baseline_model
from utils.baselines import get_baseline_similarity


def generate_synthetic_data(n_samples: int = 100, n_support: int = 10, 
                           n_query: int = 20, task_type: str = 'classification'):
    """
    Generate synthetic SMILES data for testing.
    
    Args:
        n_samples: Total number of samples
        n_support: Number of support samples
        n_query: Number of query samples
        task_type: 'classification' or 'regression'
        
    Returns:
        Tuple of (support_smiles, support_labels, query_smiles, query_labels)
    """
    # Simple synthetic SMILES for testing
    base_smiles = [
        'CCO', 'CC(C)O', 'CCCO', 'CC(C)(C)O', 'CCCCO',
        'c1ccccc1', 'c1ccccc1O', 'c1ccccc1C', 'c1ccccc1CO', 'c1ccccc1CC',
        'CC(=O)O', 'CCC(=O)O', 'CC(=O)C', 'CCC(=O)C', 'CC(=O)CC',
        'CCN', 'CCNC', 'CCNCC', 'CC(N)C', 'CCC(N)C'
    ]
    
    # Extend to n_samples by cycling
    smiles_list = []
    for i in range(n_samples):
        smiles_list.append(base_smiles[i % len(base_smiles)])
    
    # Generate labels
    if task_type == 'classification':
        labels = np.random.randint(0, 2, n_samples)
    else:
        labels = np.random.randn(n_samples) * 10 + 50
    
    # Split into support and query
    support_smiles = smiles_list[:n_support]
    support_labels = labels[:n_support]
    query_smiles = smiles_list[n_support:n_support + n_query]
    query_labels = labels[n_support:n_support + n_query]
    
    return support_smiles, support_labels, query_smiles, query_labels


def benchmark_baseline(baseline_type: str, support_smiles: List[str],
                      support_labels: np.ndarray, query_smiles: List[str],
                      query_labels: np.ndarray, task_type: str = 'classification') -> Dict[str, Any]:
    """
    Benchmark a single baseline method.
    
    Args:
        baseline_type: Type of baseline
        support_smiles: Support SMILES
        support_labels: Support labels
        query_smiles: Query SMILES
        query_labels: Query labels
        task_type: 'classification' or 'regression'
        
    Returns:
        Dictionary with results
    """
    print(f"\nBenchmarking {baseline_type}...")
    
    results = {
        'baseline_type': baseline_type,
        'task_type': task_type,
        'n_support': len(support_smiles),
        'n_query': len(query_smiles)
    }
    
    try:
        start_time = time.time()
        
        # For string-based methods, use similarity-based prediction
        if baseline_type in ['tfidf', 'bow', 'ngram', 'edit', 'jaro']:
            similarity_calc = get_baseline_similarity(baseline_type)
            
            # Fit if needed
            if hasattr(similarity_calc, 'fit'):
                similarity_calc.fit(support_smiles + query_smiles)
            
            # Predict using nearest neighbor
            predictions = []
            for query_smiles_single in query_smiles:
                similarities = []
                for support_smiles_single, support_label in zip(support_smiles, support_labels):
                    sim = similarity_calc.similarity(query_smiles_single, support_smiles_single)
                    similarities.append((sim, support_label))
                
                # Use label of most similar support sample
                similarities.sort(reverse=True)
                predictions.append(similarities[0][1])
            
            predictions = np.array(predictions)
        
        # For ML-based methods
        elif baseline_type in ['ngram-svm', 'ngram-svr', 'logistic', 'random-forest']:
            if baseline_type == 'ngram-svm':
                model = NGramMLBaseline(model_type='svm', task_type='classification')
            elif baseline_type == 'ngram-svr':
                model = NGramMLBaseline(model_type='svr', task_type='regression')
            elif baseline_type == 'logistic':
                model = NGramMLBaseline(model_type='logistic', task_type='classification')
            elif baseline_type == 'random-forest':
                model = NGramMLBaseline(model_type='random_forest', task_type=task_type)
            
            model.fit(support_smiles, support_labels)
            predictions = model.predict(query_smiles)
        
        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")
        
        training_time = time.time() - start_time
        
        # Calculate metrics
        if task_type == 'classification':
            accuracy = np.mean(predictions == query_labels)
            results['accuracy'] = float(accuracy)
            results['error_rate'] = float(1 - accuracy)
        else:
            mse = np.mean((predictions - query_labels) ** 2)
            rmse = np.sqrt(mse)
            results['mse'] = float(mse)
            results['rmse'] = float(rmse)
        
        results['training_time'] = training_time
        results['status'] = 'success'
        
    except Exception as e:
        results['status'] = 'failed'
        results['error'] = str(e)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Baseline benchmark tool')
    parser.add_argument('--baseline-type', type=str, default='all',
                       choices=['tfidf', 'bow', 'ngram', 'edit', 'jaro',
                               'ngram-svm', 'ngram-svr', 'logistic', 'random-forest', 'all'],
                       help='Type of baseline to benchmark')
    parser.add_argument('--input', type=str, default=None,
                       help='Input file with SMILES and labels (not implemented - using synthetic data)')
    parser.add_argument('--output', type=str, default='baseline_results.json',
                       help='Output file for results')
    parser.add_argument('--support-size', type=int, default=10,
                       help='Number of support samples')
    parser.add_argument('--query-size', type=int, default=20,
                       help='Number of query samples')
    parser.add_argument('--task-type', type=str, default='classification',
                       choices=['classification', 'regression'],
                       help='Task type')
    
    args = parser.parse_args()
    
    # Generate synthetic data
    print(f"Generating synthetic data ({args.support_size} support, {args.query_size} query)...")
    support_smiles, support_labels, query_smiles, query_labels = generate_synthetic_data(
        n_support=args.support_size,
        n_query=args.query_size,
        task_type=args.task_type
    )
    
    # Determine which baselines to run
    if args.baseline_type == 'all':
        baselines = ['tfidf', 'bow', 'ngram', 'edit', 'jaro']
        if args.task_type == 'classification':
            baselines.extend(['ngram-svm', 'logistic', 'random-forest'])
        else:
            baselines.extend(['ngram-svr', 'random-forest'])
    else:
        baselines = [args.baseline_type]
    
    # Run benchmarks
    all_results = []
    for baseline in baselines:
        result = benchmark_baseline(
            baseline,
            support_smiles,
            support_labels,
            query_smiles,
            query_labels,
            task_type=args.task_type
        )
        all_results.append(result)
        
        # Print result
        if result['status'] == 'success':
            print(f"  Time: {result['training_time']:.3f}s")
            if args.task_type == 'classification':
                print(f"  Accuracy: {result['accuracy']:.3f}")
            else:
                print(f"  RMSE: {result['rmse']:.3f}")
        else:
            print(f"  Failed: {result['error']}")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    
    # Print summary table
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Baseline':<20} {'Time (s)':<12} {'Metric':<20} {'Status':<10}")
    print("-"*80)
    
    for result in all_results:
        baseline = result['baseline_type']
        time_str = f"{result.get('training_time', 0):.3f}"
        
        if result['status'] == 'success':
            if args.task_type == 'classification':
                metric = f"Acc: {result['accuracy']:.3f}"
            else:
                metric = f"RMSE: {result['rmse']:.3f}"
            status = "✓"
        else:
            metric = "-"
            status = "✗"
        
        print(f"{baseline:<20} {time_str:<12} {metric:<20} {status:<10}")
    
    print("="*80)


if __name__ == '__main__':
    main()
