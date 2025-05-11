import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from src.FGWAlign import FGWAlign
from torch_geometric.datasets import TUDataset
import argparse
import warnings
import os.path as osp


def get_ad_split_TU(args, fold=5):
    DS = args.dataset
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'dataset', DS)
    print('Loading dataset from', path)
    dataset = TUDataset(path, name=DS)
    data_list = []
    label_list = []

    for data in dataset:
        data_list.append(data)
        label_list.append(data.y.item())

    kfd = StratifiedKFold(n_splits=fold, random_state=0, shuffle=True)

    splits = []
    for k, (train_index, test_index) in enumerate(kfd.split(data_list, label_list)):
        splits.append((train_index, test_index))

    return splits
warnings.filterwarnings("ignore")


# ---------------------- Parameter Parsing ----------------------
parser = argparse.ArgumentParser(description='5-Fold Graph Anomaly Detection with FGW-Full')
parser.add_argument('--dataset', type=str, default='NCI1')
parser.add_argument('--num_trials', type=int, default=5)  # 5-fold cross validation
parser.add_argument('--device', type=str, default="cpu")  # Device parameter (supports cpu/cuda:0 etc.)
args = parser.parse_args()

# ---------------------- Data Loading Function ----------------------
def get_processed_data(args, split):
    """Returns lists of preprocessed training and test graphs for the current fold"""
    DS = args.dataset
    path = f"./dataset/{DS}"
    
    # Load dataset (handle special graph types, extend as needed)
    dataset = TUDataset(path, name=DS)
    data_list = [data.to(args.device) for data in dataset]  # Move to target device
    
    # Split and filter training set (assume label 0 is normal)
    train_index, test_index = split
    data_train_ = [data_list[i] for i in train_index]
    data_test = [data_list[i] for i in test_index]
    data_train = [data for data in data_train_ if data.y != 0]  # Keep anomalous samples for training?
    
    # Relabel (training set as 0, test anomalies as 1)
    for idx, data in enumerate(data_train):
        data.y = 0
        data['idx'] = idx  # Unique index for training graphs
    for data in data_test:
        data.y = 1 if data.y == 0 else 0  # Original label 0 is anomalous
    
    return data_train, data_test

if __name__ == "__main__":
    splits = get_ad_split_TU(args, fold=args.num_trials)  # 5 folds
    auc_scores = []  # Store ROC-AUC scores for each fold
    
    for fold_idx in range(args.num_trials):
        print(f"\n============= Running Fold {fold_idx+1}/{args.num_trials} =============")
        # ---------------------- Load data for current fold ----------------------
        current_split = splits[fold_idx]
        data_train, data_test = get_processed_data(args, current_split)
        print(f"Current fold training set size: {len(data_train)}, test set size: {len(data_test)}")
        
        # ---------------------- Precompute adjacency matrices and labels for training graphs ----------------------
        train_precomputed = []
        for train_graph in data_train:
            # Extract node labels (assume x is one-hot) and adjacency matrix
            g2_labels = train_graph.x.argmax(1)
            g2_adj = torch.sparse_coo_tensor(
                train_graph.edge_index, torch.ones(train_graph.edge_index.shape[1]),
                (g2_labels.shape[0], g2_labels.shape[0])
            ).to_dense()  # Convert to dense adjacency matrix
            train_precomputed.append((g2_adj, g2_labels))
            
        # ---------------------- Test the current fold's test set ----------------------
        all_ad_pred = []
        all_ad_true = []
        
        for test_graph in data_test:
            # Preprocess fixed information for test graph (compute only once)
            g1_labels = test_graph.x.argmax(1)
            g1_adj = torch.sparse_coo_tensor(
                test_graph.edge_index, torch.ones(test_graph.edge_index.shape[1]),
                (g1_labels.shape[0], g1_labels.shape[0])
            ).to_dense()  # Test graph adjacency matrix
            
            # Calculate FGW distance between test graph and all training graphs, use minimum as anomaly score
            geds = [
                FGWAlign(g1_adj, g2_adj, g1_labels, g2_labels, 
                        patience=1, topk=5, sparse=True, device=args.device)[0]
                for g2_adj, g2_labels in train_precomputed
            ]
            min_ged = min(geds)  # Minimum distance as anomaly score (smaller distance means more normal)
            
            all_ad_pred.append(min_ged)
            all_ad_true.append(test_graph.y)
            
        # ---------------------- Evaluate ROC-AUC for current fold ----------------------
        current_auc = roc_auc_score(all_ad_true, all_ad_pred)
        auc_scores.append(current_auc)
        print(f"Fold {fold_idx+1} ROC-AUC: {current_auc:.4f}")
        
    # ---------------------- Summarize 5-fold results ----------------------
    auc_mean = np.mean(auc_scores)
    auc_std = np.std(auc_scores)
    print(f"\n============= 5-Fold Cross Validation Results Summary =============")
    print(f"ROC-AUC per fold: {[round(score, 4) for score in auc_scores]}")
    print(f"Average ROC-AUC: {auc_mean:.4f} ± {auc_std:.4f}")