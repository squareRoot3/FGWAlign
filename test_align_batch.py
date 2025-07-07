import os
import sys
import warnings
import torch
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple

# Add GABoost to the system path
sys.path.append('GABoost')

# Import custom modules
from src.FGWAlign import *
from utils import *
from GABoost.algorithm.SCMN import SCMN
from GABoost.metrics import *
from graphloader import * 

def setup_environment(cuda_device: str):
    """
    Sets up the environment by configuring CUDA devices and suppressing warnings.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    warnings.filterwarnings("ignore")
    print(f"Using CUDA device: {cuda_device}")

def save_mappings(res: torch.Tensor, dataset: str, changeline: str, sample: str, filename: str = 'mappings.txt'):
    """
    Saves the alignment mappings to a text file specific to the graph pair.
    """
    # Create a unique filename for each graph pair
    unique_filename = f'mappings_{changeline}_{sample}.txt'
    with open('results/' + unique_filename, 'w') as f:
        for i in range(res.shape[0]):
            if dataset == 'douban':
                # Assuming 1-based indexing for 'douban'
                f.write(f'{res[i].argmax().item() + 1}\t{i + 1}\n')
            else:
                # Assuming 0-based indexing for other datasets
                f.write(f'{res[i].argmax().item()}\t{i}\n')
    print(f"Mappings saved to {unique_filename}")

def compute_metrics(initial_alignment, gt, graph0, graph1):
    """
    Computes evaluation metrics for the given alignment.
    """
    ACC = accuracy(initial_alignment, gt)
    MAP = mean_average_precision(gt, graph0, graph1, dis='GABoost', predict_alignment=initial_alignment)
    EC = edge_correctness(initial_alignment, graph0, graph1)
    ICS = induced_conserved_structure(initial_alignment, graph0, graph1)
    return ACC, MAP, EC, ICS

def print_metrics(title: str, ACC: float, MAP: float, EC: float, ICS: float):
    """
    Prints the evaluation metrics with a given title.
    """
    print(f'---------------{title} Alignment---------------')
    print(f'Alignment accuracy (ACC) = {ACC:.4f}')
    print(f'Mean average precision (MAP) = {MAP:.4f}')
    print(f'Edge correctness (EC) = {EC:.4f}')
    print(f'Induced conserved structure (ICS) = {ICS:.4f}\n')

def parse_arguments():
    """
    Parses command-line arguments for CUDA device, dataset, and run_GABoost option.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='megadiff_changes',
        choices=['megadiff_changes'],
        help='Name of the dataset to use (default: megadiff_changes)'
    )
    parser.add_argument(
        '--cuda_device',
        type=str,
        default='0',
        help='CUDA device ID to use (default: 0)'
    )
    parser.add_argument(
        '--run_GABoost',
        action='store_true',
        help='Flag to indicate whether to run the final alignment using GABoost (default: False)'
    )
    return parser.parse_args()

def get_graph_pairs(dataset: str) -> List[Tuple[str, str]]:
    """
    Returns a list of (changeline, sample) tuples for the megadiff_changes dataset.
    For other datasets, returns a single tuple with empty strings.
    """
    if dataset == 'megadiff_changes':
        changelines = ['5changelines', '10changelines', '20changelines', '30changelines', '40changelines']
        samples = ['sample1', 'sample2']
        graph_pairs = []
        for changeline in changelines:
            for sample in samples:
                graph_pairs.append((changeline, sample))
        return graph_pairs
    else:
        # For other datasets, assume single graph pair
        return [('', '')]

def process_graph_pair(dataset: str, changeline: str, sample: str, device: torch.device, run_GABoost: bool):
    """
    Processes a single graph pair: loads graphs, performs alignment, computes metrics.
    Returns the metrics as a tuple.
    """
    # Define dataset paths
    if dataset == 'megadiff_changes':
        dataset_path = os.path.join('GABoost', 'dataset', dataset, 'perm-megadiff-1', changeline, sample)
    else:
        dataset_path = os.path.join('GABoost', 'dataset', dataset)
        # Add additional path handling for other datasets if needed

    true_matching_path = os.path.join(dataset_path, 'true_matching')
    left_node_file = os.path.join(dataset_path, 'left_node_file')
    left_edge_file = os.path.join(dataset_path, 'left_edge_file')
    right_node_file = os.path.join(dataset_path, 'right_node_file')
    right_edge_file = os.path.join(dataset_path, 'right_edge_file')

    # Read graphs
    graph0 = read_graph(left_node_file, left_edge_file)
    graph1 = read_graph(right_node_file, right_edge_file)

    # Convert graphs to adjacency matrices
    Aadj = torch.tensor(nx.adjacency_matrix(graph0).toarray()).float()
    Aadj = torch.maximum(Aadj, Aadj.T)
    Badj = torch.tensor(nx.adjacency_matrix(graph1).toarray()).float()
    Badj = torch.maximum(Badj, Badj.T)

    # Extract node types
    node_type0 = nx.get_node_attributes(graph0, 'type')
    node_type1 = nx.get_node_attributes(graph1, 'type')
    diff = 1 if dataset == 'douban' else 0

    # Create label tensors
    Alabel = torch.tensor([node_type0.get(i, 0) for i in range(len(node_type0))]).float()
    Blabel = torch.tensor([node_type1.get(i, 0) for i in range(len(node_type1))]).float()

    # Handle 'douban' dataset if needed
    if dataset == 'douban':
        Alabel = Alabel.argmax(dim=1)
        Blabel = Blabel.argmax(dim=1)

    # Compute optimal transport alignment
    distance, alignmat = FGWAlign(
        Aadj, Badj, Alabel, Blabel,
        device=device,
        patience=10,
        topk=1,
        sparse=True,
        alpha=1,
    )
    res = alignmat.T.to_dense().cpu()

    # Save mappings
    save_mappings(res, dataset, changeline, sample)

    # Read ground truth and initial alignment
    gt = read_matching(true_matching_path)
    initial_alignment = read_matching(f'results/mappings_{changeline}_{sample}.txt')

    # Compute and print initial metrics
    ACC, MAP, EC, ICS = compute_metrics(initial_alignment, gt, graph0, graph1)
    print_metrics(f"Initial ({changeline}_{sample})", ACC, MAP, EC, ICS)

    if run_GABoost:
        # Initialize and run GABoost algorithm
        alg = GABoost(graph0, graph1, initial_alignment)
        output_alignment = alg.get_matching(64)

        # Compute and print final metrics
        ACC_final, MAP_final, EC_final, ICS_final = compute_metrics(output_alignment, gt, graph0, graph1)
        print_metrics(f"Final ({changeline}_{sample})", ACC_final, MAP_final, EC_final, ICS_final)
    else:
        ACC_final = MAP_final = EC_final = ICS_final = None
        print(f"Final alignment skipped for ({changeline}_{sample}).\n")

    return (ACC, MAP, EC, ICS, ACC_final, MAP_final, EC_final, ICS_final)

def main():
    # Parse command-line arguments
    args = parse_arguments()
    dataset = args.dataset
    cuda_device = args.cuda_device
    run_GABoost = args.run_GABoost

    # Setup environment
    setup_environment(cuda_device=cuda_device)

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Get list of graph pairs to process
    graph_pairs = get_graph_pairs(dataset)

    # Initialize lists to store metrics
    ACC_initial_list = []
    MAP_initial_list = []
    EC_initial_list = []
    ICS_initial_list = []
    ACC_final_list = []
    MAP_final_list = []
    EC_final_list = []
    ICS_final_list = []

    # Process each graph pair
    for changeline, sample in graph_pairs:
        pair_name = f"{changeline}_{sample}" if dataset == 'megadiff_changes' else "single_pair"
        print(f"Processing graph pair: {pair_name}" if dataset == 'megadiff_changes' else "Processing graph pair")
        metrics = process_graph_pair(dataset, changeline, sample, device, run_GABoost)
        if dataset == 'megadiff_changes':
            ACC_initial, MAP_initial, EC_initial, ICS_initial, ACC_final, MAP_final, EC_final, ICS_final = metrics
            ACC_initial_list.append(ACC_initial)
            MAP_initial_list.append(MAP_initial)
            EC_initial_list.append(EC_initial)
            ICS_initial_list.append(ICS_initial)
            if run_GABoost and ACC_final is not None:
                ACC_final_list.append(ACC_final)
                MAP_final_list.append(MAP_final)
                EC_final_list.append(EC_final)
                ICS_final_list.append(ICS_final)
        else:
            # For other datasets, adjust accordingly
            pass

    # Compute and print average metrics for megadiff_changes
    if dataset == 'megadiff_changes':
        num_pairs = len(ACC_initial_list)
        avg_ACC_initial = sum(ACC_initial_list) / num_pairs
        avg_MAP_initial = sum(MAP_initial_list) / num_pairs
        avg_EC_initial = sum(EC_initial_list) / num_pairs
        avg_ICS_initial = sum(ICS_initial_list) / num_pairs

        print('=============== Average Metrics for megadiff_changes ===============')
        print('Initial Alignment:')
        print(f'Average ACC = {avg_ACC_initial:.4f}')
        print(f'Average MAP = {avg_MAP_initial:.4f}')
        print(f'Average EC  = {avg_EC_initial:.4f}')
        print(f'Average ICS = {avg_ICS_initial:.4f}\n')

        if run_GABoost and ACC_final_list:
            avg_ACC_final = sum(ACC_final_list) / len(ACC_final_list)
            avg_MAP_final = sum(MAP_final_list) / len(MAP_final_list)
            avg_EC_final = sum(EC_final_list) / len(EC_final_list)
            avg_ICS_final = sum(ICS_final_list) / len(ICS_final_list)

            print('Final Alignment:')
            print(f'Average ACC = {avg_ACC_final:.4f}')
            print(f'Average MAP = {avg_MAP_final:.4f}')
            print(f'Average EC  = {avg_EC_final:.4f}')
            print(f'Average ICS = {avg_ICS_final:.4f}\n')
        else:
            print('Final Alignment was not executed. Average metrics for final alignment are unavailable.\n')
    else:
        pass

if __name__ == '__main__':
    main()
