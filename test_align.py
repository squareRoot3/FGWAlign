import os
import sys
import warnings
import torch
import argparse

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

def save_mappings(res: torch.Tensor, dataset: str, filename: str = 'mappings.txt'):
    """
    Saves the alignment mappings to a text file.
    """
    with open(filename, 'w') as f:
        for i in range(res.shape[0]):
            if dataset == 'douban':
                # Assuming 1-based indexing for 'douban'
                f.write(f'{res[i].argmax().item() + 1}\t{i + 1}\n')
            else:
                # Assuming 0-based indexing for other datasets
                f.write(f'{res[i].argmax().item()}\t{i}\n')
    print(f"Mappings saved to {filename}")

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
    Parses command-line arguments for CUDA device and dataset.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='douban',
        help='Name of the dataset to use (default: douban)'
    )
    parser.add_argument(
        '--cuda_device',
        type=str,
        default='0',
        help='CUDA device ID to use'
    )
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()
    dataset = args.dataset
    cuda_device = args.cuda_device

    # Define dataset paths
    dataset_path = os.path.join('GABoost', 'dataset', dataset)

    true_matching_path = os.path.join(dataset_path, 'true_matching')
    left_node_file = os.path.join(dataset_path, 'left_node_file')
    left_edge_file = os.path.join(dataset_path, 'left_edge_file')
    right_node_file = os.path.join(dataset_path, 'right_node_file')
    right_edge_file = os.path.join(dataset_path, 'right_edge_file')
    graph0 = read_graph(left_node_file, left_edge_file)
    graph1 = read_graph(right_node_file, right_edge_file)

    Aadj = torch.tensor(nx.adjacency_matrix(graph0).toarray()).float()
    Aadj = torch.maximum(Aadj, Aadj.T)
    Badj = torch.tensor(nx.adjacency_matrix(graph1).toarray()).float()
    Badj = torch.maximum(Badj, Badj.T)
    node_type0 = nx.get_node_attributes(graph0, 'type')
    node_type1 = nx.get_node_attributes(graph1, 'type')
    diff = 1 if dataset == 'douban' else 0

    Alabel = torch.tensor([node_type0[i+diff] for i in range(len(node_type0))]).float()
    Blabel = torch.tensor([node_type1[i+diff] for i in range(len(node_type1))]).float()
    # Alabel = torch.zeros_like(Alabel)
    # Blabel = torch.zeros_like(Blabel)
        
    # Setup environment
    setup_environment(cuda_device=cuda_device)

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load data
    # Aadj, Badj, Alabel, Blabel, ground_truth = load_data(dataset, device)

    # Compute optimal transport alignment
    distance, alignmat = FGWAlign(
        Aadj, Badj, Alabel, Blabel,
        device=device,
        patience=10,
        topk=1,
        sparse=True,
        alpha=100,
    )
    res = alignmat.T.to_dense().cpu()

    # Save mappings
    save_mappings(res, dataset, filename='results/mappings.txt')


    # Read matchings and graphs
    gt = read_matching(true_matching_path)
    initial_alignment = read_matching('results/mappings.txt')


    # Compute and print initial metrics
    ACC, MAP, EC, ICS = compute_metrics(initial_alignment, gt, graph0, graph1)
    print_metrics("Initial", ACC, MAP, EC, ICS)

    # Initialize and run GABoost algorithm
    alg = GABoost(graph0, graph1, initial_alignment)
    output_alignment = alg.get_matching(64)

    # Compute and print final metrics
    ACC, MAP, EC, ICS = compute_metrics(output_alignment, gt, graph0, graph1)
    print_metrics("Final", ACC, MAP, EC, ICS)

if __name__ == '__main__':
    main()
