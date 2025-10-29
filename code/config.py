g_batch_size = 128
d_batch_size = 128
lambda_gen = 1e-5
lambda_dis = 1e-5
lr_gen = 1e-4
lr_dis = 1e-4
n_epoch = 100
sig = 1.0
label_smooth = 0.0
d_epoch = 15
g_epoch = 5
n_emb = 128
dataset = 'cora'
experiment = 'link_prediction'
train_file = '../data/%s/train_0.5' % dataset
test_file = '../data/%s/test_0.5' % dataset
pretrain_ckpt = ''
pretrain_dis_node_emb = []
pretrain_gen_node_emb = []
save = True
save_path = '../results/%s/%s/' % (experiment, dataset)
save_last = True
verbose = 1
log = True

# Determine n_node dynamically
import os
def get_n_node():
    """Determine the number of nodes dynamically from data."""
    # Fallback default in case data files aren't accessible
    default_n_node = 0
    
    # If train file doesn't exist, use default
    if not os.path.exists(train_file):
        print(f"Warning: Train file {train_file} not found, using default n_node={default_n_node}")
        return default_n_node
        
    # Read train file to count nodes
    try:
        from utils import read_graph
        _, actual_n_node, node_list, _, _ = read_graph(train_file)
        
        # Ensure node_list is a list
        if not isinstance(node_list, list) and not isinstance(node_list, set):
            print(f"Warning: node_list is not iterable (type: {type(node_list)}), converting to list")
            # If node_list is an int, create a list from range(0, node_list)
            if isinstance(node_list, int):
                node_list = list(range(node_list))
            else:
                # If it's something else unexpected, use an empty list
                node_list = []
        
        # Check test file for any additional nodes
        test_file_0 = test_file + '_0'
        if os.path.exists(test_file_0):
            test_nodes = set()
            with open(test_file_0, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        test_nodes.add(int(parts[0]))
                        test_nodes.add(int(parts[1]))
            
            # Find max node ID considering both train and test
            if node_list and test_nodes:
                max_node_id = max(max(node_list), max(test_nodes))
            elif node_list:
                max_node_id = max(node_list)
            elif test_nodes:
                max_node_id = max(test_nodes)
            else:
                max_node_id = default_n_node - 1
            return max_node_id + 1
        
        # If no test file, use max ID from train + 1
        max_node_id = max(node_list) if node_list else default_n_node - 1
        return max(actual_n_node, max_node_id + 1)
        
    except Exception as e:
        print(f"Error determining n_node: {str(e)}")
        print(f"Using default n_node={default_n_node}")
        return default_n_node

# Set n_node dynamically
n_node = get_n_node()
print(f"Using n_node = {n_node}")

import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")