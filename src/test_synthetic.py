import argparse
import datetime
import os
import pickle
import random
import time
import networkx as nx
import numpy as np
import torch
from typing import Any, Dict, List, Tuple
from memory_profiler import memory_usage
from FGWAlign import *
from baselines import *


def set_seed(seed: int = 0) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run GED experiments.")
    parser.add_argument(
        "--method",
        type=str,
        default="FGWAlign",
        choices=["FGWAlign", "ipfp", "spectral", "rrwm", "astar", "astar-beam"],
        help="Name of the method.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        help="Name of the dataset.",
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=16384,
        help="Maximum node size in the experiment.",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=200,
        help="Number of test pairs to compute in the experiment.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Exploration patience T in FGWAlign.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=1,
        help="Diverse projection candidates K in FGWAlign.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model.",
    )
    parser.add_argument(
        "--reg",
        type=float,
        default=0.0,
        help="Weight of the sparse regularization term in FGWAlign.",
    )
    return parser.parse_args()


def load_ged_list(dataset_path: str) -> List[Tuple[Any, Any, int]]:
    """Load GED list from a pickle file."""
    with open(dataset_path, 'rb') as f:
        ged_list = pickle.load(f)
    return ged_list


def calculate_relative_error(pred: int, gt: int) -> float:
    """Calculate the relative error."""
    return abs(pred - gt) / gt if gt != 0 else 0.0


def cal_pk(num: int, pre: List[int], gt: List[int]) -> float:
    """Calculate the p@k metric."""
    tmp = sorted(zip(gt, pre), key=lambda x: x[1])
    beta = sorted([(p[1], p[0], i) for i, p in enumerate(tmp)], key=lambda x: x[0])
    ans = sum(1 for i in range(num) if beta[i][2] < num)
    return ans / num if num > 0 else 0.0


def measure_memory(func, *args, **kwargs) -> Tuple[List[float], Any]:
    """Measure memory usage of a function."""
    mem_usage = memory_usage((func, args, kwargs), max_iterations=1, interval=0.1)
    result = func(*args, **kwargs)
    return mem_usage, result


def process_FGWAlign(
    g1: nx.Graph,
    g2: nx.Graph,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[int, float]:
    """Process GED using FGWAlign method."""
    n1, n2 = g1.number_of_nodes(), g2.number_of_nodes()
    g1_labels = torch.tensor([g1.nodes[i]['label'] for i in range(n1)], device=device)
    g2_labels = torch.tensor([g2.nodes[i]['label'] for i in range(n2)], device=device)

    g1_edges = torch.tensor(list(g1.edges)).T.to(device)
    g1_edges = torch.cat([g1_edges, g1_edges[[1, 0]]], dim=1)
    g2_edges = torch.tensor(list(g2.edges)).T.to(device)
    g2_edges = torch.cat([g2_edges, g2_edges[[1, 0]]], dim=1)

    g1_adj = torch.sparse_coo_tensor(
        g1_edges,
        torch.ones(g1_edges.size(1), device=device),
        (n1, n1)
    ).to_dense()
    g2_adj = torch.sparse_coo_tensor(
        g2_edges,
        torch.ones(g2_edges.size(1), device=device),
        (n2, n2)
    ).to_dense()

    pred, _ = FGWAlign(
        g1_adj,
        g2_adj,
        g1_labels,
        g2_labels,
        patience=args.patience,
        reg=args.reg,
        device=device,
        sparse=True,
        topk=args.topk,
    )
    max_gpu_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == 'cuda' else 0.0
    torch.cuda.reset_peak_memory_stats(device) if device.type == 'cuda' else None

    return pred, max_gpu_memory


def process_other_methods(
    g1: Any,
    g2: Any,
    args: argparse.Namespace,
) -> Tuple[int, float]:
    """Process GED using non-FGWAlign methods."""
    mem_usage, (pred, _) = measure_memory(cal_ged_pygm, g1, g2, args, format='synthetic')
    memory_consumed = max(mem_usage) - min(mem_usage) if mem_usage else 0.0
    return pred, memory_consumed


def save_results_to_txt(results: Dict[int, Dict[str, List[Any]]], filepath: str) -> None:
    """Save the results dictionary to a text file in a readable format."""
    with open(filepath, 'w') as f:
        f.write("Final Results:\n")
        for ns, metrics in sorted(results.items()):
            avg_acc = np.mean(metrics["acc"]) if metrics["acc"] else 0.0
            avg_error = np.mean(metrics["error"]) if metrics["error"] else 0.0
            avg_time = np.mean(metrics["time"]) if metrics["time"] else 0.0
            max_memory = np.max(metrics["memory"]) if metrics["memory"] else 0.0
            f.write(f'Node size {ns}:\n')
            f.write(f'  Accuracy: {avg_acc:.4f}\n')
            f.write(f'  Error: {avg_error:.4f}\n')
            f.write(f'  Time: {avg_time:.4f} seconds\n')
            f.write(f'  Memory: {max_memory:.2f} MB\n')
            f.write('----------------------------------\n')
    print(f"Results saved to {filepath}")


def main() -> None:
    """Main function to run GED experiments."""
    set_seed()
    args = parse_arguments()
    print(f"Arguments: {args}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("Using CPU")

    results: Dict[int, Dict[str, List[Any]]] = {}
    dataset_path = os.path.join('dataset', f"{args.dataset}.pkl")
    ged_list = load_ged_list(dataset_path)

    for idx, (g1, g2, gt) in enumerate(ged_list, 1):
        ns = g1.number_of_nodes()
        if ns not in results:
            results[ns] = {'acc': [], 'time': [], 'error': [], 'memory': []}

        if ns > args.max_size or len(results[ns]['acc']) > args.max_pairs:
            continue

        start_time = time.time()

        try:
            if args.method == 'FGWAlign':
                pred, memory = process_FGWAlign(g1, g2, args, device)
            else:
                pred, memory = process_other_methods(g1, g2, args)
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue

        end_time = time.time()
        acc = int(pred == gt)
        time_cost = end_time - start_time
        relative_error = calculate_relative_error(pred, gt)

        results[ns]['acc'].append(acc)
        results[ns]['time'].append(time_cost)
        results[ns]['error'].append(relative_error)
        results[ns]['memory'].append(memory)

        print(
            f"Sample {idx}: Node size {ns}, GED: {gt}, Pred: {pred}, "
            f"Relative Error: {relative_error:.4f}, Memory: {memory:.2f} MB"
        )

    print("\nFinal Results:")
    for ns, metrics in sorted(results.items()):
        avg_acc = np.mean(metrics["acc"]) if metrics["acc"] else 0.0
        avg_error = np.mean(metrics["error"]) if metrics["error"] else 0.0
        avg_time = np.mean(metrics["time"]) if metrics["time"] else 0.0
        max_memory = np.max(metrics["memory"]) if metrics["memory"] else 0.0
        print(f'Node size {ns}:')
        print(f'  Accuracy: {avg_acc:.4f}')
        print(f'  Error: {avg_error:.4f}')
        print(f'  Time: {avg_time:.4f} seconds')
        print(f'  Memory: {max_memory:.2f} MB')
        print('----------------------------------')

    timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M")
    results_filename = f"{args.method}_{args.dataset}_{timestamp}.txt"
    results_path = os.path.join('results', results_filename)
    os.makedirs('results', exist_ok=True)
    save_results_to_txt(results, results_path)


if __name__ == "__main__":
    main()
