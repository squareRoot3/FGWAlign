import networkx as nx
import random
import argparse
import pickle
import time
import json
import torch
import numpy as np
from collections import defaultdict

# Set random seeds for reproducibility
def set_random_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_random_seeds(0)

def shuffle_graph(G):
    """
    Shuffle node labels to anonymize the graph.
    """
    nodes = list(G.nodes())
    shuffled_nodes = nodes.copy()
    random.shuffle(shuffled_nodes)
    mapping = {node: shuffled_nodes[i] for i, node in enumerate(nodes)}
    shuffled_G = nx.relabel_nodes(G, mapping)
    return shuffled_G

def drop_node_ged(G, ratio=0.05):
    """
    Remove a fraction of nodes from the graph and compute a simple GED.
    """
    drop_node_num = max(1, int(ratio * G.number_of_nodes()))
    drop_nodes = random.sample(list(G.nodes()), drop_node_num)
    new_G = G.copy()
    new_G.remove_nodes_from(drop_nodes)
    
    # Simple GED: number of removed nodes + number of removed edges
    removed_edges = G.number_of_edges() - new_G.number_of_edges()
    ged = drop_node_num + removed_edges
    
    return new_G, ged

def change_label_ged(G, ratio=0.05, label_range=5):
    """
    Change labels of a fraction of nodes and add a fraction of edges.
    """
    new_G = G.copy()
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    change_label_num = max(1, int(ratio * num_nodes))
    add_edges_num = max(0, int(ratio * num_edges))
    
    # Avoid making the graph too dense
    max_possible_edges = num_nodes * (num_nodes - 1) // 2
    if add_edges_num + num_edges > max_possible_edges:
        add_edges_num = max_possible_edges - num_edges
    
    # Change labels
    changed_nodes = random.sample(list(G.nodes()), change_label_num)
    label_changes = 0
    for node in changed_nodes:
        old_label = G.nodes[node].get('label', 0)
        new_label = random.randint(0, label_range - 1)
        while new_label == old_label:
            new_label = random.randint(0, label_range - 1)
        new_G.nodes[node]['label'] = new_label
        label_changes += 1
    
    # Add edges
    added_edges = 0
    attempts = 0
    max_attempts = add_edges_num * 10  # Prevent infinite loops
    while added_edges < add_edges_num and attempts < max_attempts:
        u, v = random.sample(list(G.nodes()), 2)
        if not new_G.has_edge(u, v):
            new_G.add_edge(u, v)
            added_edges += 1
        attempts += 1
    
    # GED: number of label changes + number of edge additions
    ged = label_changes + added_edges
    
    return new_G, ged

def generate_synthetic_graphs(node_sizes, graph_num, corrupt_ratio, label_range=5):
    """
    Generate synthetic graphs using various models and apply modifications.
    """
    ged_list = []
    graph_types = [
        'powerlaw_cluster',
        'erdos_renyi',
        'barabasi_albert',
        'gaussian_partition'
    ]

    for ns in node_sizes:
        for i in range(graph_num):
            print(f"Generating graphs for node size {ns}, graph {i+1}/{graph_num}")
            # Generate different types of graphs
            graphs = []
            # Powerlaw Cluster Graph
            ave_degree = random.randint(5, 10)
            G_plc = nx.powerlaw_cluster_graph(n=ns, m=ave_degree, p=0.1)
            graphs.append(G_plc)
            
            # Erdős-Rényi Graph
            p_er = ave_degree / ns
            G_er = nx.erdos_renyi_graph(n=ns, p=p_er)
            graphs.append(G_er)
            
            # Barabási-Albert Graph
            G_ba = nx.barabasi_albert_graph(n=ns, m=ave_degree)
            graphs.append(G_ba)
            
            # Gaussian Random Partition Graph
            s = max(1, ns // 5)  # Ensure s is at least 1
            v = 0.1
            p_in = 3 * ave_degree / ns
            p_out = ave_degree / ns
            G_grp = nx.gaussian_random_partition_graph(n=ns, s=s, v=v, p_in=p_in, p_out=p_out)
            graphs.append(G_grp)
            
            for G in graphs:
                print(f"Original Graph: Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()}")
                # Assign random labels to nodes
                for node in G.nodes():
                    G.nodes[node]['label'] = random.randint(0, label_range - 1)
                
                # Apply modifications
                G_dropped, ged_drop = drop_node_ged(G, ratio=corrupt_ratio)
                G_dropped_shuffled = shuffle_graph(G_dropped)
                ged_list.append({
                    'original_graph': G,
                    'modified_graph': G_dropped_shuffled,
                    'modification': 'drop_node',
                    'ged': ged_drop
                })
                
                G_modified, ged_change = change_label_ged(G, ratio=corrupt_ratio, label_range=label_range)
                G_modified_shuffled = shuffle_graph(G_modified)
                ged_list.append({
                    'original_graph': G,
                    'modified_graph': G_modified_shuffled,
                    'modification': 'change_label',
                    'ged': ged_change
                })
    
    return ged_list

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate synthetic graphs with modifications.")
    parser.add_argument('--node_sizes', type=int, nargs='+', default=[32,64,128,256,512,1024,2048,4096,8192,16384],
                        help='List of node sizes for synthetic graphs.')
    parser.add_argument('--graph_num', type=int, default=25, help='Number of graphs per node size and graph type.')
    parser.add_argument('--corrupt_ratio', type=float, default=0.25, help='Ratio of nodes/edges to corrupt.')
    parser.add_argument('--label_range', type=int, default=5, help='Number of distinct labels for nodes.')
    parser.add_argument('--output', type=str, default='synthetic_graphs.pkl', help='Output pickle file name.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    start_time = time.time()
    ged_list = generate_synthetic_graphs(
        node_sizes=args.node_sizes,
        graph_num=args.graph_num,
        corrupt_ratio=args.corrupt_ratio,
        label_range=args.label_range
    )
    end_time = time.time()
    print(f"Generated {len(ged_list)} modified graphs in {end_time - start_time:.2f} seconds.")
    
    # Save to pickle
    with open(args.output, 'wb') as f:
        pickle.dump(ged_list, f)
    print(f"Saved synthetic graphs to {args.output}")

if __name__ == '__main__':
    main()
