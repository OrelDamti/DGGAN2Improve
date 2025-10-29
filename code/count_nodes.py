import os
import sys
import config
from utils import read_graph

def count_dataset_nodes():
    """Count the actual number of nodes in the dataset"""
    
    # Print current configuration
    print(f"Current config.n_node: {config.n_node}")
    print(f"Train file: {config.train_file}")
    
    # Check if train file exists
    if not os.path.exists(config.train_file):
        print(f"ERROR: Train file {config.train_file} does not exist!")
        return
    
    # Count lines in train file
    with open(config.train_file, 'r') as f:
        edge_count = len(f.readlines())
    print(f"Total edges in train file: {edge_count}")
    
    # Use the read_graph function to determine actual node count
    graph, n_node, node_list, node_list_s, egs = read_graph(config.train_file)
    
    # Get the actual highest node ID
    max_node_id = max(max(node_list)) if node_list else -1
    
    print(f"\nActual number of nodes: {n_node}")
    print(f"Highest node ID: {max_node_id}")
    
    # Check nodes in test file as well
    test_file = config.test_file + '_0'
    if os.path.exists(test_file):
        test_nodes = set()
        with open(test_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    test_nodes.add(int(parts[0]))
                    test_nodes.add(int(parts[1]))
        
        print(f"\nUnique nodes in test file: {len(test_nodes)}")
        print(f"Highest test node ID: {max(test_nodes) if test_nodes else -1}")
        
        # Find nodes that are in test but not in train
        test_only = test_nodes - set(node_list)
        if test_only:
            print(f"WARNING: {len(test_only)} nodes appear in test but not train!")
            print(f"Example node IDs from test only: {list(test_only)[:5]}")
    
    # Recommend correct n_node value
    suggested_n_node = max(max_node_id + 1, n_node)
    
    print(f"\nRECOMMENDATION: Set n_node = {suggested_n_node} in config.py")
    
    return suggested_n_node

if __name__ == "__main__":
    count_dataset_nodes() 