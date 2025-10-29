import numpy as np
import tensorflow as tf
import random
import os
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from dggan import Model
import config
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

def repeated_evaluation_of_positive_links(test_ratio=0.3, checkpoint_path=None, visualize=True, top_k=100, k_iterations=5):
    """
    Evaluates model's ability to reconstruct positive links using the generator.
    
    Args:
        test_ratio: Ratio of positive examples to remove and try to reconstruct (default 0.3)
        checkpoint_path: Path to a model checkpoint (optional)
        visualize: Whether to visualize results with plots (default True)
        top_k: Number of top predictions to consider for each node (default 100)
        k_iterations: Number of times to repeat the evaluation with different edge selections (default 1)
        
    Returns:
        Dictionary with evaluation metrics across all iterations
    """
    print(f"Starting link reconstruction evaluation with {k_iterations} iterations. v4.1")
    # Load the test data file
    test_file = config.test_file + '_0'
    all_test_links = []
    
    print(f"Reading test data from {test_file}")
    with open(test_file, 'r') as f:
        for line in f:
            source, target, label = map(int, line.strip().split())
            all_test_links.append((source, target, label))
    
    # Filter to get only positive links from test data
    positive_links = [(s, t, l) for s, t, l in all_test_links if l == 1]
    
    print(f"Total test links: {len(all_test_links)}")
    print(f"Positive test links: {len(positive_links)}")
    
    # Initialize the model
    model = Model()
    
    if checkpoint_path:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model.checkpoint.restore(checkpoint_path)
    else:
        raise ValueError("Checkpoint path is required")
    
    # Get unique nodes from all test links
    all_nodes = set()
    for source, target, _ in all_test_links:
        all_nodes.add(source)
        all_nodes.add(target)
    n_nodes = len(all_nodes)
    print(f"Total unique nodes in test data: {n_nodes}")
    
    # Get embeddings (do this once for all iterations)
    print("Getting node embeddings...")
    embedding_matrix = model.discriminator.node_embedding_matrix.numpy()
    
    # Store results across iterations
    all_results = []
    iteration_metrics = {
        'hits_at_k': [],
        'hits_at_1': [],  # Track first-rank accuracy specifically
        'mrr': [],
        'precision_at_k': [],
        'total_predictions': [],
        'removed_links': [],  # Track number of removed links per iteration
        'remaining_links': []  # Track number of remaining known links per iteration
    }
    
    # Run k iterations
    for iteration in range(k_iterations):
        print(f"\n--- Iteration {iteration + 1}/{k_iterations} ---")
        
        # Use different random seed for each iteration
        random.seed(42 + iteration)
        random.shuffle(positive_links)
        split_idx = int((1-test_ratio) * len(positive_links))
        
        # Keep 70% as known links, 30% to reconstruct
        known_links = positive_links[:split_idx]
        to_reconstruct = positive_links[split_idx:]
        
        print(f"Known positive links: {len(known_links)}")
        print(f"Positive links to reconstruct: {len(to_reconstruct)}")
        
        # Create sets for faster lookup
        to_reconstruct_set = set([(link[0], link[1]) for link in to_reconstruct])
        
        # Create a set of all source nodes from links to reconstruct
        test_sources = set([link[0] for link in to_reconstruct])
        
        # Track metrics for this iteration
        hits_at_k = 0
        hits_at_1 = 0  # Track first-rank hits specifically
        mrr = 0.0  # Mean Reciprocal Rank
        precision_at_k = 0.0
        total_predictions = 0
        
        # For visualization
        hit_ranks = []
        
        print("Evaluating reconstruction for each source node...")
        for source_node in tqdm(test_sources):
            # Skip if source node is not in our embedding matrix
            if source_node >= embedding_matrix[0].shape[0]:
                continue
                
            source_embedding = embedding_matrix[0][source_node]
            
            # Calculate similarity with all possible target nodes
            similarities = []
            valid_targets = []
            
            for i in range(min(embedding_matrix[1].shape[0], n_nodes)):
                target_embedding = embedding_matrix[1][i]
                similarity = np.dot(source_embedding, target_embedding)
                similarities.append(similarity)
                valid_targets.append(i)
                
            # Sort targets by similarity score (descending)
            sorted_indices = np.argsort(similarities)[::-1]
            sorted_targets = [valid_targets[i] for i in sorted_indices]
            
            # Get actual targets for this source from the links to reconstruct
            actual_targets = [target for s, target, _ in to_reconstruct if s == source_node]
            
            if not actual_targets:
                continue
                
            # Consider only top k predictions
            top_k_targets = sorted_targets[:top_k]
            
            # Check if any actual targets are in top k predictions
            found = False
            for actual_target in actual_targets:
                if actual_target in top_k_targets:
                    hits_at_k += 1
                    found = True
                    
                    # Calculate rank of the hit
                    rank = top_k_targets.index(actual_target) + 1
                    hit_ranks.append(rank)
                    
                    # Check if it's a first-rank hit
                    if rank == 1:
                        hits_at_1 += 1
                    
                    # Add to MRR
                    mrr += 1.0 / rank
                    break
            
            # Count correct predictions in top k
            correct_in_topk = sum(1 for t in top_k_targets if (source_node, t) in to_reconstruct_set)
            precision_at_k += correct_in_topk / len(top_k_targets)
            
            total_predictions += 1
        
        # Calculate final metrics for this iteration
        if total_predictions > 0:
            hits_at_k_ratio = hits_at_k / total_predictions
            hits_at_1_ratio = hits_at_1 / total_predictions
            mrr_score = mrr / total_predictions
            precision_at_k_score = precision_at_k / total_predictions
        else:
            hits_at_k_ratio = 0
            hits_at_1_ratio = 0
            mrr_score = 0
            precision_at_k_score = 0
        
        # Store metrics for this iteration
        iteration_metrics['hits_at_k'].append(hits_at_k_ratio)
        iteration_metrics['hits_at_1'].append(hits_at_1_ratio)
        iteration_metrics['mrr'].append(mrr_score)
        iteration_metrics['precision_at_k'].append(precision_at_k_score)
        iteration_metrics['total_predictions'].append(total_predictions)
        iteration_metrics['removed_links'].append(len(to_reconstruct))
        iteration_metrics['remaining_links'].append(len(known_links))
        
        # Print detailed results for this iteration
        print(f"\n--- Iteration {iteration + 1} Detailed Results ---")
        print(f"1. Removed links (edges to reconstruct): {len(to_reconstruct)}")
        print(f"2. Remaining known links (true edges left): {len(known_links)}")
        print(f"3. Hits@1 (first-rank accuracy): {hits_at_1_ratio:.4f} ({hits_at_1}/{total_predictions})")
        print(f"   Hits@{top_k}: {hits_at_k_ratio:.4f} ({hits_at_k}/{total_predictions})")
        print(f"   MRR: {mrr_score:.4f}")
        print(f"   Precision@{top_k}: {precision_at_k_score:.4f}")
        
        # Store detailed results for this iteration
        iteration_result = {
            'iteration': iteration + 1,
            'hits_at_k': hits_at_k_ratio,
            'hits_at_1': hits_at_1_ratio,
            'mrr': mrr_score,
            'precision_at_k': precision_at_k_score,
            'hit_ranks': hit_ranks,
            'total_predictions': total_predictions,
            'removed_links': len(to_reconstruct),
            'remaining_links': len(known_links)
        }
        all_results.append(iteration_result)
    
    # Calculate overall statistics
    print(f"\n=== OVERALL RESULTS ACROSS {k_iterations} ITERATIONS ===")
    avg_hits = np.mean(iteration_metrics['hits_at_k'])
    std_hits = np.std(iteration_metrics['hits_at_k'])
    avg_hits_1 = np.mean(iteration_metrics['hits_at_1'])
    std_hits_1 = np.std(iteration_metrics['hits_at_1'])
    avg_mrr = np.mean(iteration_metrics['mrr'])
    std_mrr = np.std(iteration_metrics['mrr'])
    avg_precision = np.mean(iteration_metrics['precision_at_k'])
    std_precision = np.std(iteration_metrics['precision_at_k'])
    avg_removed = np.mean(iteration_metrics['removed_links'])
    avg_remaining = np.mean(iteration_metrics['remaining_links'])
    
    print(f"Average removed links per iteration: {avg_removed:.1f}")
    print(f"Average remaining known links per iteration: {avg_remaining:.1f}")
    print(f"Hits@1 (First-rank accuracy): {avg_hits_1:.4f} ± {std_hits_1:.4f}")
    print(f"Hits@{top_k}: {avg_hits:.4f} ± {std_hits:.4f}")
    print(f"MRR: {avg_mrr:.4f} ± {std_mrr:.4f}")
    print(f"Precision@{top_k}: {avg_precision:.4f} ± {std_precision:.4f}")
    
    # Visualize results across iterations
    if visualize and k_iterations > 1:
        plt.figure(figsize=(15, 16))
        
        # Plot performance across iterations
        iterations = list(range(1, k_iterations + 1))
        
        plt.subplot(4, 1, 1)
        plt.plot(iterations, iteration_metrics['hits_at_1'], 'o-', linewidth=2, markersize=6, color='red')
        plt.title('Hits@1 (First-rank Accuracy) Across Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Hits@1')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=avg_hits_1, color='darkred', linestyle='--', alpha=0.7, label=f'Average: {avg_hits_1:.4f}')
        plt.legend()
        
        plt.subplot(4, 1, 2)
        plt.plot(iterations, iteration_metrics['hits_at_k'], 'o-', linewidth=2, markersize=6)
        plt.title(f'Hits@{top_k} Across Iterations')
        plt.xlabel('Iteration')
        plt.ylabel(f'Hits@{top_k}')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=avg_hits, color='r', linestyle='--', alpha=0.7, label=f'Average: {avg_hits:.4f}')
        plt.legend()
        
        plt.subplot(4, 1, 3)
        plt.plot(iterations, iteration_metrics['mrr'], 'o-', linewidth=2, markersize=6, color='orange')
        plt.title('MRR Across Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('MRR')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=avg_mrr, color='r', linestyle='--', alpha=0.7, label=f'Average: {avg_mrr:.4f}')
        plt.legend()
        
        plt.subplot(4, 1, 4)
        plt.plot(iterations, iteration_metrics['precision_at_k'], 'o-', linewidth=2, markersize=6, color='green')
        plt.title(f'Precision@{top_k} Across Iterations')
        plt.xlabel('Iteration')
        plt.ylabel(f'Precision@{top_k}')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=avg_precision, color='r', linestyle='--', alpha=0.7, label=f'Average: {avg_precision:.4f}')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Additional plot: Box plot for distribution visualization
        if k_iterations >= 5:  # Only show box plot if we have enough iterations
            plt.figure(figsize=(15, 4))
            
            metrics_data = [iteration_metrics['hits_at_1'],
                          iteration_metrics['hits_at_k'], 
                          iteration_metrics['mrr'], 
                          iteration_metrics['precision_at_k']]
            labels = ['Hits@1', f'Hits@{top_k}', 'MRR', f'Precision@{top_k}']
            
            plt.boxplot(metrics_data, labels=labels)
            plt.title('Distribution of Metrics Across All Iterations')
            plt.ylabel('Score')
            plt.grid(True, alpha=0.3)
            plt.show()

    # Prepare visualization data for machine interpretation
    histogram_data = {}
    cumulative_hits_data = {}
    
    # Handle visualization for single iteration case
    if k_iterations == 1 and visualize and all_results:
        hit_ranks = all_results[0]['hit_ranks']
        if hit_ranks:
            plt.figure(figsize=(15, 10))
            
            # Plot rank distribution
            plt.subplot(2, 1, 1)
            n_bins = min(50, len(hit_ranks))
            counts, bins, patches = plt.hist(hit_ranks, bins=n_bins, alpha=0.7)
            plt.title('Distribution of Hit Ranks')
            plt.xlabel('Rank')
            plt.ylabel('Count')
            
            # Store histogram data for machine interpretation
            histogram_data = {
                'counts': counts.tolist(),
                'bins': bins.tolist(),
                'bin_centers': [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
            }
            
            # Plot cumulative hits
            plt.subplot(2, 1, 2)
            ranks = list(range(1, min(top_k+1, max(hit_ranks)+1)))
            hits_at_ranks = [sum(1 for r in hit_ranks if r <= k) / all_results[0]['total_predictions'] for k in ranks]
            plt.plot(ranks, hits_at_ranks, '-', linewidth=2)
            plt.title('Cumulative Hits@k')
            plt.xlabel('k')
            plt.ylabel('Hits@k')
            plt.grid(True)
            
            # Store cumulative hits data for machine interpretation
            cumulative_hits_data = {
                'ranks': ranks,
                'hits_at_ranks': hits_at_ranks
            }
            
            plt.tight_layout()
            plt.show()
    elif k_iterations == 1 and all_results:  # Even if not visualizing, compute the data
        hit_ranks = all_results[0]['hit_ranks']
        if hit_ranks:
            # Compute histogram data
            n_bins = min(50, len(hit_ranks))
            counts, bins = np.histogram(hit_ranks, bins=n_bins)
            histogram_data = {
                'counts': counts.tolist(),
                'bins': bins.tolist(),
                'bin_centers': [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
            }
            
            # Compute cumulative hits data
            ranks = list(range(1, min(top_k+1, max(hit_ranks)+1)))
            hits_at_ranks = [sum(1 for r in hit_ranks if r <= k) / all_results[0]['total_predictions'] for k in ranks]
            cumulative_hits_data = {
                'ranks': ranks,
                'hits_at_ranks': hits_at_ranks
            }
    
    # Output data for machine interpretation
    print("\n=== MACHINE INTERPRETABLE DATA ===")
    if k_iterations == 1:
        print("HISTOGRAM_DATA:")
        print(f"bins: {histogram_data.get('bins', [])}")
        print(f"counts: {histogram_data.get('counts', [])}")
        print(f"bin_centers: {histogram_data.get('bin_centers', [])}")
        
        print("\nCUMULATIVE_HITS_DATA:")
        print(f"ranks: {cumulative_hits_data.get('ranks', [])}")
        print(f"hits_at_ranks: {cumulative_hits_data.get('hits_at_ranks', [])}")
    else:
        print("ITERATION_METRICS:")
        print(f"iterations: {list(range(1, k_iterations + 1))}")
        print(f"hits_at_1: {iteration_metrics['hits_at_1']}")
        print(f"hits_at_k: {iteration_metrics['hits_at_k']}")
        print(f"mrr: {iteration_metrics['mrr']}")
        print(f"precision_at_k: {iteration_metrics['precision_at_k']}")
        print(f"total_predictions: {iteration_metrics['total_predictions']}")
        print(f"removed_links: {iteration_metrics['removed_links']}")
        print(f"remaining_links: {iteration_metrics['remaining_links']}")
        
        print("\nAGGREGATE_STATISTICS:")
        print(f"avg_removed_links: {avg_removed}")
        print(f"avg_remaining_links: {avg_remaining}")
        print(f"avg_hits_at_1: {avg_hits_1}")
        print(f"std_hits_at_1: {std_hits_1}")
        print(f"avg_hits_at_k: {avg_hits}")
        print(f"std_hits_at_k: {std_hits}")
        print(f"avg_mrr: {avg_mrr}")
        print(f"std_mrr: {std_mrr}")
        print(f"avg_precision_at_k: {avg_precision}")
        print(f"std_precision_at_k: {std_precision}")
    print("=== END MACHINE INTERPRETABLE DATA ===\n")
    
    return {
        'all_iterations': all_results,
        'iteration_metrics': iteration_metrics,
        'aggregate_stats': {
            'avg_removed_links': avg_removed,
            'avg_remaining_links': avg_remaining,
            'avg_hits_at_1': avg_hits_1,
            'std_hits_at_1': std_hits_1,
            'avg_hits_at_k': avg_hits,
            'std_hits_at_k': std_hits,
            'avg_mrr': avg_mrr,
            'std_mrr': std_mrr,
            'avg_precision_at_k': avg_precision,
            'std_precision_at_k': std_precision
        },
        'k_iterations': k_iterations,
        'histogram_data': histogram_data,
        'cumulative_hits_data': cumulative_hits_data
    }

# For usage in Colab, you can call this function directly
if __name__ == "__main__":
    # Example:
    # checkpoint_path = None  # Set to None to train a new model
    # evaluate_positive_links(test_ratio=0.3, checkpoint_path=checkpoint_path, k_iterations=10)
    repeated_evaluation_of_positive_links(k_iterations=5)  # Run 5 iterations by default 