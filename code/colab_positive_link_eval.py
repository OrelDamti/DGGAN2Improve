import numpy as np
import tensorflow as tf
import random
import os
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from dggan import Model
import config
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

def evaluate_positive_links(test_ratio=0.3, checkpoint_path=None, visualize=True, top_k=100):
    """
    Evaluates model's ability to reconstruct positive links using the generator.
    
    Args:
        test_ratio: Ratio of positive examples to remove and try to reconstruct (default 0.3)
        checkpoint_path: Path to a model checkpoint (optional)
        visualize: Whether to visualize results with plots (default True)
        top_k: Number of top predictions to consider for each node (default 100)
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("Starting link reconstruction evaluation. v4.0")
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
    
    # Shuffle and split the positive links
    random.seed(42)  # For reproducibility
    random.shuffle(positive_links)
    split_idx = int((1-test_ratio) * len(positive_links))
    
    # Keep 70% as known links, 30% to reconstruct
    known_links = positive_links[:split_idx]
    to_reconstruct = positive_links[split_idx:]
    
    print(f"Known positive links: {len(known_links)}")
    print(f"Positive links to reconstruct: {len(to_reconstruct)}")
    
    # Initialize the model
    model = Model()
    
    if checkpoint_path:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model.checkpoint.restore(checkpoint_path)
    else:
        raise ValueError("Checkpoint path is required")
    
    # Create sets for faster lookup
    to_reconstruct_set = set([(link[0], link[1]) for link in to_reconstruct])
    
    # Create a set of all source nodes from links to reconstruct
    test_sources = set([link[0] for link in to_reconstruct])
    
    # Get unique nodes from all test links
    all_nodes = set()
    for source, target, _ in all_test_links:
        all_nodes.add(source)
        all_nodes.add(target)
    n_nodes = len(all_nodes)
    print(f"Total unique nodes in test data: {n_nodes}")
    
    # Get embeddings
    print("Getting node embeddings...")
    embedding_matrix = model.discriminator.node_embedding_matrix.numpy()
    
    # Track metrics
    hits_at_k = 0
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
                
                # Add to MRR
                mrr += 1.0 / rank
                break
        
        # Count correct predictions in top k
        correct_in_topk = sum(1 for t in top_k_targets if (source_node, t) in to_reconstruct_set)
        precision_at_k += correct_in_topk / len(top_k_targets)
        
        total_predictions += 1
    
    # Calculate final metrics
    if total_predictions > 0:
        hits_at_k_ratio = hits_at_k / total_predictions
        mrr = mrr / total_predictions
        precision_at_k = precision_at_k / total_predictions
    else:
        hits_at_k_ratio = 0
        mrr = 0
        precision_at_k = 0
    
    # Print results
    print("\nLink Reconstruction Evaluation Results:")
    print(f"Hits@{top_k}: {hits_at_k_ratio:.4f} ({hits_at_k}/{total_predictions})")
    print(f"MRR: {mrr:.4f}")
    print(f"Precision@{top_k}: {precision_at_k:.4f}")
    
    # Visualize results if requested
    if visualize and hit_ranks:
        plt.figure(figsize=(15, 10))
        
        # Plot rank distribution
        plt.subplot(2, 1, 1)
        plt.hist(hit_ranks, bins=min(50, len(hit_ranks)), alpha=0.7)
        plt.title('Distribution of Hit Ranks')
        plt.xlabel('Rank')
        plt.ylabel('Count')
        
        # Plot cumulative hits
        plt.subplot(2, 1, 2)
        ranks = range(1, min(top_k+1, max(hit_ranks)+1))
        hits_at_ranks = [sum(1 for r in hit_ranks if r <= k) / total_predictions for k in ranks]
        plt.plot(ranks, hits_at_ranks, '-', linewidth=2)
        plt.title('Cumulative Hits@k')
        plt.xlabel('k')
        plt.ylabel('Hits@k')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    return {
        'hits_at_k': hits_at_k_ratio,
        'mrr': mrr,
        'precision_at_k': precision_at_k,
        'hit_ranks': hit_ranks if hit_ranks else [],
        'total_predictions': total_predictions
    }

# For usage in Colab, you can call this function directly
if __name__ == "__main__":
    # Example:
    # checkpoint_path = None  # Set to None to train a new model
    # evaluate_positive_links(test_ratio=0.3, checkpoint_path=checkpoint_path)
    evaluate_positive_links() 