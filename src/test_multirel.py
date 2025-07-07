import argparse
import pickle
import time
import random
import numpy as np
import torch
from FGWAlign import FGWAlign, FGWAlign_multirel
from scipy.stats import spearmanr, kendalltau


def set_seed(seed: int = 0):
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run GED experiments.")
    parser.add_argument(
        "--method",
        type=str,
        default="FGWAlign_rel",
        choices=["FGWAlign", "FGWAlign_rel"],
        help="Name of the GED method to use.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Exploration patience T in FGWAlign.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model on (e.g., 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="dataset/AIDS/multirel_test.pkl",
        help="Path to the test dataset pickle file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional path to save the results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def calculate_precision_at_k(k: int, predictions: list, ground_truths: list) -> float:
    """
    Calculate the precision at K (PK) metric.

    Args:
        k (int): The value of K.
        predictions (list): List of predicted GED values.
        ground_truths (list): List of ground truth GED values.

    Returns:
        float: The PK metric.
    """
    paired = sorted(zip(ground_truths, predictions), key=lambda x: x[0])
    beta = sorted([(pred, gt, idx) for idx, (gt, pred) in enumerate(paired)], key=lambda x: x[0])

    correct = sum(1 for i in range(min(k, len(beta))) if beta[i][2] < k)
    return correct / k if k > 0 else 0.0


def load_test_data(dataset_path: str):
    """
    Load the test dataset from a pickle file.

    Args:
        dataset_path (str): Path to the pickle file.

    Returns:
        dict: The loaded test dataset.
    """
    try:
        with open(dataset_path, "rb") as f:
            test_dict = pickle.load(f)
        return test_dict
    except FileNotFoundError:
        print(f"Dataset file not found at path: {dataset_path}")
        exit(1)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)


def prepare_adjacency_matrix(g, num_nodes: int, num_relations: int = 3, device: torch.device = torch.device('cpu')):
    """
    Prepare a list of adjacency matrices for multi-relational graphs.

    Args:
        g: Graph object containing nodes and edges with labels.
        num_nodes (int): Number of nodes in the graph.
        num_relations (int, optional): Number of relation types. Defaults to 3.
        device (torch.device, optional): The device for tensors. Defaults to 'cpu'.

    Returns:
        list of torch.Tensor: Adjacency matrices per relation.
    """
    adj_list = [torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device) for _ in range(num_relations)]
    for edge in g.edges():
        src, dst = int(edge[0]), int(edge[1])
        label = g.edges[edge].get('label', 1)  # Default relation label 1 if not present
        if 1 <= label <= num_relations:
            adj = adj_list[label - 1]
            adj[src, dst] = 1.0
            adj[dst, src] = 1.0  # Assuming undirected graphs
        else:
            print(f"Encountered invalid relation label {label} on edge ({src}, {dst}). Skipping.")
    return adj_list


def main():
    args = parse_arguments()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f"Using device: {device}")

    mse, rho, tau, pk20, acc, time_list = [], [], [], [], [], []
    test_dict = load_test_data(args.dataset_path)

    for g1_id, pair_list in test_dict.items():
        pred_list, gt_list = [], []
        for pair in pair_list:
            g1, g2, ged = pair.get('g1'), pair.get('g2'), pair.get('ged')

            if g1 is None or g2 is None or ged is None:
                print(f"Missing data in pair {pair}. Skipping.")
                continue

            n1, n2 = g1.number_of_nodes(), g2.number_of_nodes()
            g1_labels = torch.tensor([g1.nodes[str(i)].get('label', 0) for i in range(n1)], dtype=torch.long, device=device)
            g2_labels = torch.tensor([g2.nodes[str(i)].get('label', 0) for i in range(n2)], dtype=torch.long, device=device)

            start_time = time.time()
            if args.method == 'FGWAlign':
                edge_list1 = [[int(src), int(dst)] for src, dst in g1.edges()]
                edge_list2 = [[int(src), int(dst)] for src, dst in g2.edges()]

                g1_edges = torch.tensor(edge_list1, dtype=torch.long).T.to(device)
                g1_edges = torch.cat([g1_edges, g1_edges[[1, 0]]], dim=1)  # Make undirected
                g2_edges = torch.tensor(edge_list2, dtype=torch.long).T.to(device)
                g2_edges = torch.cat([g2_edges, g2_edges[[1, 0]]], dim=1)  # Make undirected

                g1_adj = torch.sparse_coo_tensor(g1_edges, torch.ones(g1_edges.size(1), dtype=torch.float32, device=device),
                                                 (n1, n1)).to_dense()
                g2_adj = torch.sparse_coo_tensor(g2_edges, torch.ones(g2_edges.size(1), dtype=torch.float32, device=device),
                                                 (n2, n2)).to_dense()

                pred, _ = FGWAlign(
                    g1_adj, g2_adj, g1_labels, g2_labels,
                    patience=args.patience
                )

            elif args.method == 'FGWAlign_rel':
                g1_adj_list = prepare_adjacency_matrix(g1, n1, device=device)
                g2_adj_list = prepare_adjacency_matrix(g2, n2, device=device)

                pred, _ = FGWAlign_multirel(
                    g1_adj_list, g2_adj_list,
                    g1_labels, g2_labels,
                    patience=args.patience
                )
            else:
                print(f"Unknown method: {args.method}. Skipping.")
                continue

            elapsed_time = time.time() - start_time
            time_list.append(elapsed_time)

            gt_list.append(ged)
            pred_list.append(pred)
            acc.append(int(pred == ged))
            mse.append((pred - ged) ** 2)

            # Optional: print detailed information per pair
            # print(f'GED: {ged}, Pred: {pred}, Time: {elapsed_time:.4f}s')

        if pred_list and gt_list:
            rho_val, _ = spearmanr(pred_list, gt_list)
            tau_val, _ = kendalltau(pred_list, gt_list)
            rho.append(rho_val if rho_val is not None else 0.0)
            tau.append(tau_val if tau_val is not None else 0.0)
            pk20_val = calculate_precision_at_k(20, pred_list, gt_list)
            pk20.append(pk20_val)
        else:
            print(f"No valid predictions for graph id {g1_id}.")

    # Calculate average metrics
    avg_acc = np.mean(acc) if acc else 0.0
    avg_mse = np.mean(mse) if mse else 0.0
    avg_rho = np.mean(rho) if rho else 0.0
    avg_tau = np.mean(tau) if tau else 0.0
    avg_pk20 = np.mean(pk20) if pk20 else 0.0
    avg_time = np.mean(time_list) if time_list else 0.0

    result_str = (
        f'Accuracy: {avg_acc:.4f}, '
        f'MSE: {avg_mse:.4f}, '
        f'Spearman: {avg_rho:.4f}, '
        f'Kendall: {avg_tau:.4f}, '
        f'PK20: {avg_pk20:.4f}, '
        f'Time: {avg_time:.4f}s'
    )
    print(result_str)

    # Optionally, save the results to a file
    if args.output_file:
        try:
            with open(args.output_file, "w") as f:
                f.write(result_str + "\n")
            print(f"Results saved to {args.output_file}")
        except Exception as e:
            print(f"Error saving results to file: {e}")


if __name__ == "__main__":
    main()
