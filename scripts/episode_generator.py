"""Episode generator with baseline support."""

import argparse
import json
import time
from typing import List, Dict, Any
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.baseline_models import NGramMLBaseline
from utils.baselines import get_baseline_similarity


def generate_episode(n_support: int = 5, n_query: int = 10, 
                    task_type: str = 'classification') -> Dict[str, Any]:
    """
    Generate a single few-shot learning episode.
    
    Args:
        n_support: Number of support samples
        n_query: Number of query samples
        task_type: 'classification' or 'regression'
        
    Returns:
        Episode dictionary
    """
    # Simple synthetic SMILES for testing
    base_smiles = [
        'CCO', 'CC(C)O', 'CCCO', 'CC(C)(C)O', 'CCCCO',
        'c1ccccc1', 'c1ccccc1O', 'c1ccccc1C', 'c1ccccc1CO', 'c1ccccc1CC',
        'CC(=O)O', 'CCC(=O)O', 'CC(=O)C', 'CCC(=O)C', 'CC(=O)CC',
        'CCN', 'CCNC', 'CCNCC', 'CC(N)C', 'CCC(N)C'
    ]
    
    total_samples = n_support + n_query
    smiles_list = []
    for i in range(total_samples):
        smiles_list.append(base_smiles[i % len(base_smiles)])
    
    # Generate labels
    if task_type == 'classification':
        labels = np.random.randint(0, 2, total_samples)
    else:
        labels = np.random.randn(total_samples) * 10 + 50
    
    episode = {
        'support': {
            'smiles': smiles_list[:n_support],
            'labels': labels[:n_support].tolist()
        },
        'query': {
            'smiles': smiles_list[n_support:],
            'labels': labels[n_support:].tolist()
        }
    }
    
    return episode


def run_baseline(episode: Dict[str, Any], baseline_type: str,
                task_type: str = 'classification') -> Dict[str, Any]:
    """
    Run baseline method on episode.
    
    Args:
        episode: Episode dictionary
        baseline_type: Type of baseline
        task_type: 'classification' or 'regression'
        
    Returns:
        Results dictionary
    """
    support_smiles = episode['support']['smiles']
    support_labels = np.array(episode['support']['labels'])
    query_smiles = episode['query']['smiles']
    query_labels = np.array(episode['query']['labels'])
    
    start_time = time.time()
    
    # String-based methods
    if baseline_type in ['tfidf', 'bow', 'ngram', 'edit', 'jaro']:
        similarity_calc = get_baseline_similarity(baseline_type)
        
        if hasattr(similarity_calc, 'fit'):
            similarity_calc.fit(support_smiles + query_smiles)
        
        predictions = []
        for query_smiles_single in query_smiles:
            similarities = []
            for support_smiles_single, support_label in zip(support_smiles, support_labels):
                sim = similarity_calc.similarity(query_smiles_single, support_smiles_single)
                similarities.append((sim, support_label))
            
            similarities.sort(reverse=True)
            predictions.append(similarities[0][1])
        
        predictions = np.array(predictions)
    
    # ML-based methods
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
    
    inference_time = time.time() - start_time
    
    # Calculate metrics
    results = {
        'baseline_type': baseline_type,
        'predictions': predictions.tolist(),
        'inference_time': inference_time
    }
    
    if task_type == 'classification':
        accuracy = np.mean(predictions == query_labels)
        results['accuracy'] = float(accuracy)
    else:
        mse = np.mean((predictions - query_labels) ** 2)
        rmse = np.sqrt(mse)
        results['mse'] = float(mse)
        results['rmse'] = float(rmse)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Episode generator with baseline support')
    parser.add_argument('--n-episodes', type=int, default=10,
                       help='Number of episodes to generate')
    parser.add_argument('--support-size', type=int, default=5,
                       help='Number of support samples per episode')
    parser.add_argument('--query-size', type=int, default=10,
                       help='Number of query samples per episode')
    parser.add_argument('--task-type', type=str, default='classification',
                       choices=['classification', 'regression'],
                       help='Task type')
    parser.add_argument('--baseline', action='store_true',
                       help='Enable baseline methods')
    parser.add_argument('--baseline-type', type=str, default='tfidf',
                       choices=['tfidf', 'bow', 'ngram', 'edit', 'jaro',
                               'ngram-svm', 'ngram-svr', 'logistic', 'random-forest', 'all'],
                       help='Type of baseline to use')
    parser.add_argument('--compare-with-baselines', action='store_true',
                       help='Compare with multiple baseline methods')
    parser.add_argument('--output', type=str, default='episodes.json',
                       help='Output file for episodes')
    
    args = parser.parse_args()
    
    # Generate episodes
    print(f"Generating {args.n_episodes} episodes...")
    episodes = []
    for i in range(args.n_episodes):
        episode = generate_episode(
            n_support=args.support_size,
            n_query=args.query_size,
            task_type=args.task_type
        )
        episodes.append(episode)
    
    # Run baselines if requested
    if args.baseline or args.compare_with_baselines:
        if args.compare_with_baselines:
            # Run multiple baselines
            baseline_types = ['tfidf', 'bow', 'ngram', 'edit', 'jaro']
            if args.task_type == 'classification':
                baseline_types.extend(['ngram-svm', 'logistic'])
            else:
                baseline_types.append('ngram-svr')
        elif args.baseline_type == 'all':
            baseline_types = ['tfidf', 'bow', 'ngram', 'edit', 'jaro']
            if args.task_type == 'classification':
                baseline_types.extend(['ngram-svm', 'logistic'])
            else:
                baseline_types.append('ngram-svr')
        else:
            baseline_types = [args.baseline_type]
        
        print(f"\nRunning baselines: {', '.join(baseline_types)}")
        
        # Run baselines on each episode
        for i, episode in enumerate(episodes):
            print(f"\nEpisode {i+1}/{args.n_episodes}")
            episode['baseline_results'] = {}
            
            for baseline_type in baseline_types:
                try:
                    result = run_baseline(episode, baseline_type, args.task_type)
                    episode['baseline_results'][baseline_type] = result
                    
                    if args.task_type == 'classification':
                        print(f"  {baseline_type}: Acc={result['accuracy']:.3f}, Time={result['inference_time']:.3f}s")
                    else:
                        print(f"  {baseline_type}: RMSE={result['rmse']:.3f}, Time={result['inference_time']:.3f}s")
                
                except Exception as e:
                    print(f"  {baseline_type}: Failed - {str(e)}")
                    episode['baseline_results'][baseline_type] = {'error': str(e)}
    
    # Save episodes
    with open(args.output, 'w') as f:
        json.dump(episodes, f, indent=2)
    
    print(f"\nEpisodes saved to {args.output}")
    
    # Print summary if baselines were run
    if args.baseline or args.compare_with_baselines:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        # Aggregate results across episodes
        baseline_stats = {}
        for episode in episodes:
            if 'baseline_results' in episode:
                for baseline_type, result in episode['baseline_results'].items():
                    if 'error' not in result:
                        if baseline_type not in baseline_stats:
                            baseline_stats[baseline_type] = []
                        
                        if args.task_type == 'classification':
                            baseline_stats[baseline_type].append(result['accuracy'])
                        else:
                            baseline_stats[baseline_type].append(result['rmse'])
        
        # Print average performance
        print(f"{'Baseline':<20} {'Avg Metric':<15}")
        print("-"*35)
        for baseline_type, scores in baseline_stats.items():
            avg_score = np.mean(scores)
            if args.task_type == 'classification':
                print(f"{baseline_type:<20} Acc: {avg_score:.3f}")
            else:
                print(f"{baseline_type:<20} RMSE: {avg_score:.3f}")
        print("="*80)


if __name__ == '__main__':
    main()
