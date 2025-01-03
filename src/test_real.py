import sys
import time
import json
import torch
import random
import os
import numpy as np
import networkx as nx
import datetime
import argparse
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr, kendalltau
from os.path import basename, isfile
from FGWAlign import *
from baselines import *
from tqdm import tqdm
from glob import glob


def optimal_unique_assignment(align_mat):
    """
    Assign each row in align_mat to a unique column optimally based on alignment scores using the Hungarian algorithm.

    Parameters:
    align_mat (torch.Tensor): A 2D tensor of shape (num_rows, num_cols) representing alignment scores.

    Returns:
    List[int]: A list where the ith element is the assigned column index for the ith row.
    """
    cost_matrix = -align_mat.cpu().numpy()  # Convert to numpy and negate for maximization
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    assignment = {row: col for row, col in zip(row_ind, col_ind)}
    num_rows = align_mat.shape[0]
    pre_permute = [assignment.get(row, -1) for row in range(num_rows)]
    return pre_permute


def load_ged(ged_dict, data_location='.', dataset_name='AIDS', file_name='TaGED.json'):
    '''
    list(tuple)
    ged = [(id_1, id_2, ged_value, ged_nc, ged_in, ged_ie, [best_node_mapping])]

    id_1 and id_2 are the IDs of a graph pair, e.g., the ID of 4.json is 4.
    The given graph pairs satisfy that n1 <= n2.

    ged_value = ged_nc + ged_in + ged_ie
    (ged_nc, ged_in, ged_ie) is the type-aware ged following the setting of TaGSim.
    ged_nc: the number of node relabeling
    ged_in: the number of node insertions/deletions
    ged_ie: the number of edge insertions/deletions

    [best_node_mapping] contains 10 best matching at most.
    best_node_mapping is a list of length n1: u in g1 -> best_node_mapping[u] in g2

    return dict()
    ged_dict[(id_1, id_2)] = ((ged_value, ged_nc, ged_in, ged_ie), best_node_mapping_list)
    '''
    path = "{}dataset/{}/{}".format(data_location, dataset_name, file_name)
    TaGED = json.load(open(path, 'r'))
    for (id_1, id_2, ged_value, ged_nc, ged_in, ged_ie, mappings) in TaGED:
        ta_ged = (ged_value, ged_nc, ged_in, ged_ie)
        ged_dict[(id_1, id_2)] = (ta_ged, mappings)

def sorted_nicely(l):
    """
    Sort file names in a fancy way.
    The numbers in file names are extracted and converted from str into int first,
    so file names can be sorted based on int comparison.
    :param l: A list of file names:str.
    :return: A nicely sorted file name list.
    """

    def tryint(s):
        try:
            return int(s)
        except:
            return s

    import re
    def alphanum_key(s):
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    return sorted(l, key=alphanum_key)

def get_file_paths(dir, file_format='json'):
    """
    Return all file paths with file_format under dir.
    :param dir: Input path.
    :param file_format: The suffix name of required files.
    :return paths: The paths of all required files.
    """
    dir = dir.rstrip('/')
    paths = sorted_nicely(glob(dir + '/*.' + file_format))
    return paths

def iterate_get_graphs(dir, file_format):
    """
    Read networkx (dict) graphs from all .gexf (.json) files under dir.
    :param dir: Input path.
    :param file_format: The suffix name of required files.
    :return graphs: Networkx (dict) graphs.
    """
    assert file_format in ['gexf', 'json', 'onehot', 'anchor']
    graphs = []
    for file in get_file_paths(dir, file_format):
        gid = int(basename(file).split('.')[0])
        if file_format == 'gexf':
            g = nx.read_gexf(file)
            g.graph['gid'] = gid
            if not nx.is_connected(g):
                raise RuntimeError('{} not connected'.format(gid))
        elif file_format == 'json':
            # g is a dict
            g = json.load(open(file, 'r'))
            g['gid'] = gid
        elif file_format in ['onehot', 'anchor']:
            # g is a list of onehot labels
            g = json.load(open(file, 'r'))
        graphs.append(g)
    return graphs

def load_labels(data_location, dataset_name):
    path = data_location + "dataset/" + dataset_name + "/labels.json"
    global_labels = json.load(open(path, 'r'))
    features = iterate_get_graphs(data_location + "dataset/" + dataset_name + "/train", "onehot") \
             + iterate_get_graphs(data_location + "dataset/" + dataset_name + "/test", "onehot")
    print('Load one-hot label features (dim = {}) of {}.'.format(len(global_labels), dataset_name))
    return global_labels, features

def load_all_graphs(data_location, dataset_name):
    graphs = iterate_get_graphs(data_location + "dataset/" + dataset_name + "/train", "json")
    train_num = len(graphs)
    graphs += iterate_get_graphs(data_location + "dataset/" + dataset_name + "/test", "json")
    test_num = len(graphs) - train_num
    val_num = test_num
    train_num -= val_num
    return train_num, val_num, test_num, graphs


class GED_Tester(object):
    def __init__(self, args):
        self.args = args
        self.load_data_time = 0.0
        self.to_torch_time = 0.0
        self.results = []
        self.load_data()
        self.transfer_data_to_torch()
        self.delta_graphs = [None] * len(self.graphs)
        self.gen_delta_graphs()
        self.init_graph_pairs()

    def load_data(self):
        """
        Load graphs, ged and labels if needed.
        self.ged: dict-dict, ged['graph_id_1']['graph_id_2'] stores the ged value.
        """
        t1 = time.time()
        dataset_name = self.args.dataset
        self.train_num, self.val_num, self.test_num, self.graphs = load_all_graphs('', dataset_name)
        print("Load {} graphs. ({} for training)".format(len(self.graphs), self.train_num))

        self.number_of_labels = 0
        if dataset_name in ['AIDS']:
            self.global_labels, self.features = load_labels('', dataset_name)
            self.number_of_labels = len(self.global_labels)
        if self.number_of_labels == 0:
            self.number_of_labels = 1
            self.features = []
            for g in self.graphs:
                self.features.append([[0.0] for u in range(g['n'])])

        ged_dict = dict()
        load_ged(ged_dict, '', dataset_name, 'TaGED.json')
        self.ged_dict = ged_dict
        print("Load ged dict.")
        t2 = time.time()
        self.load_data_time = t2 - t1

    def transfer_data_to_torch(self):
        """
        Transfer loaded data to torch.
        """
        t1 = time.time()

        self.edge_index = []
        for g in self.graphs:
            edge = g['graph']
            edge = edge + [[y, x] for x, y in edge]
            edge = edge + [[x, x] for x in range(g['n'])]
            edge = torch.tensor(edge).t().long()
            self.edge_index.append(edge)

        self.features = [torch.tensor(x).float() for x in self.features]
        print("Feature shape of 1st graph:", self.features[0].shape)

        n = len(self.graphs)
        mapping = [[None for i in range(n)] for j in range(n)]
        ged = [[(0., 0., 0., 0.) for i in range(n)] for j in range(n)]
        gid = [g['gid'] for g in self.graphs]
        self.gid = gid
        self.gn = [g['n'] for g in self.graphs]
        self.gm = [g['m'] for g in self.graphs]
        for i in range(n):
            mapping[i][i] = torch.eye(self.gn[i], dtype=torch.float)
            for j in range(i + 1, n):
                id_pair = (gid[i], gid[j])
                n1, n2 = self.gn[i], self.gn[j]
                if id_pair not in self.ged_dict:
                    id_pair = (gid[j], gid[i])
                    n1, n2 = n2, n1
                if id_pair not in self.ged_dict:
                    ged[i][j] = ged[j][i] = None
                    mapping[i][j] = mapping[j][i] = None
                else:
                    ta_ged, gt_mappings = self.ged_dict[id_pair]
                    ged[i][j] = ged[j][i] = ta_ged
                    mapping_list = [[0 for y in range(n2)] for x in range(n1)]
                    for gt_mapping in gt_mappings:
                        for x, y in enumerate(gt_mapping):
                            mapping_list[x][y] = 1
                    mapping_matrix = torch.tensor(mapping_list).float()
                    mapping[i][j] = mapping[j][i] = mapping_matrix
        self.ged = ged
        self.mapping = mapping

        t2 = time.time()
        self.to_torch_time = t2 - t1

    @staticmethod
    def delta_graph(g, f):
        new_data = dict()

        n = g['n']
        permute = list(range(n))
        random.shuffle(permute)
        mapping = torch.sparse_coo_tensor((list(range(n)), permute), [1.0] * n, (n, n)).to_dense()

        edge = g['graph']
        edge_set = set()
        for x, y in edge:
            edge_set.add((x, y))
            edge_set.add((y, x))

        random.shuffle(edge)
        m = len(edge)
        ged = random.randint(1, 5) if n <= 20 else random.randint(1, 10)
        del_num = min(m, random.randint(0, ged))
        edge = edge[:(m - del_num)]  # the last del_num edges in edge are removed
        add_num = ged - del_num
        if (add_num + m) * 2 > n * (n - 1):
            add_num = n * (n - 1) // 2 - m
        cnt = 0
        while cnt < add_num:
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
            if (x != y) and (x, y) not in edge_set:
                edge_set.add((x, y))
                edge_set.add((y, x))
                cnt += 1
                edge.append([x, y])
        assert len(edge) == m - del_num + add_num
        new_data["n"] = n
        new_data["m"] = len(edge)

        new_edge = [[permute[x], permute[y]] for x, y in edge]
        new_edge = new_edge + [[y, x] for x, y in new_edge]  # add reverse edges
        new_edge = new_edge + [[x, x] for x in range(n)]  # add self-loops

        new_edge = torch.tensor(new_edge).t().long()

        feature2 = torch.zeros(f.shape)
        for x, y in enumerate(permute):
            feature2[y] = f[x]

        new_data["permute"] = permute
        new_data["mapping"] = mapping
        ged = del_num + add_num
        new_data["ta_ged"] = (ged, 0, 0, ged)
        new_data["edge_index"] = new_edge
        new_data["features"] = feature2
        return new_data

    def gen_delta_graphs(self):
        k = self.args.num_testing_graphs
        for i, g in enumerate(self.graphs):
            # Do not generate delta graphs for small graphs.
            if g['n'] <= 10:
                continue
            # gen k delta graphs
            f = self.features[i]
            self.delta_graphs[i] = [self.delta_graph(g, f) for j in range(k)]

    def gen_edit_path(self, data, permute):
        n1, n2 = data["n1"], data["n2"]
        raw_edges_1, raw_edges_2 = data["edge_index_1"].t().tolist(), data["edge_index_2"].t().tolist()
        raw_f1, raw_f2 = data["features_1"].tolist(), data["features_2"].tolist()
        # print(n1, n2, len(permute))
        assert len(permute) == n1
        assert len(raw_f1) == n1 and len(raw_f2) == n2 and len(raw_f1[0]) == len(raw_f2[0])

        edges_1 = set()
        for (u, v) in raw_edges_1:
            pu, pv = permute[u], permute[v]
            if pu <= pv:
                edges_1.add((pu, pv))

        edges_2 = set()
        for (u, v) in raw_edges_2:
            if u <= v:
                edges_2.add((u, v))

        edit_edges = edges_1 ^ edges_2

        f1 = []
        num_label = len(raw_f1[0])
        for f in raw_f1:
            for j in range(num_label):
                if f[j] > 0:
                    f1.append(j)
                    break
        f2 = []
        for f in raw_f2:
            for j in range(num_label):
                if f[j] > 0:
                    f2.append(j)
                    break

        relabel_nodes = set()
        if self.args.dataset == 'AIDS':
            for (u, v) in enumerate(permute):
                if f1[u] != f2[v]:
                    relabel_nodes.add((v, f1[u]))

        return edit_edges, relabel_nodes

    def check_pair(self, i, j):
        if i == j:
            return (0, i, j)
        id1, id2 = self.gid[i], self.gid[j]
        if (id1, id2) in self.ged_dict:
            return (0, i, j)
        elif (id2, id1) in self.ged_dict:
            return (0, j, i)
        else:
            return None

    def init_graph_pairs(self):
        random.seed(1)

        self.training_graphs = []
        self.val_graphs = []
        self.testing_graphs = []
        self.testing2_graphs = []

        train_num = self.train_num
        val_num = train_num + self.val_num
        test_num = len(self.graphs)

        dg = self.delta_graphs
        for i in range(train_num):
            if self.gn[i] <= 10:
                for j in range(i, train_num):
                    tmp = self.check_pair(i, j)
                    if tmp is not None:
                        self.training_graphs.append(tmp)
            elif dg[i] is not None:
                k = len(dg[i])
                for j in range(k):
                    self.training_graphs.append((1, i, j))

        li = []
        for i in range(train_num):
            if self.gn[i] <= 10:
                li.append(i)

        for i in range(train_num, val_num):
            if self.gn[i] <= 10:
                random.shuffle(li)
                self.val_graphs.append((0, i, li[:self.args.num_testing_graphs]))
            elif dg[i] is not None:
                k = len(dg[i])
                self.val_graphs.append((1, i, list(range(k))))

        for i in range(val_num, test_num):
            if self.gn[i] <= 10:
                random.shuffle(li)
                self.testing_graphs.append((0, i, li[:self.args.num_testing_graphs]))
            elif dg[i] is not None:
                k = len(dg[i])
                self.testing_graphs.append((1, i, list(range(k))))
        print("Generate {} training graph pairs.".format(len(self.training_graphs)))
        print("Generate {} * {} val graph pairs.".format(len(self.val_graphs), self.args.num_testing_graphs))
        print("Generate {} * {} testing graph pairs.".format(len(self.testing_graphs), self.args.num_testing_graphs))

    @staticmethod
    def cal_pk(num, pre, gt):
        tmp = list(zip(gt, pre))
        tmp.sort()
        beta = []
        for i, p in enumerate(tmp):
            beta.append((p[1], p[0], i))
        beta.sort()
        ans = 0
        for i in range(num):
            if beta[i][2] < num:
                ans += 1
        return ans / num

    def pack_graph_pair(self, graph_pair):
        """
        Prepare the graph pair data for GedGNN model.
        :param graph_pair: (pair_type, id_1, id_2)
        :return new_data: Dictionary of Torch Tensors.
        """
        new_data = dict()

        (pair_type, id_1, id_2) = graph_pair
        if pair_type == 0:  # normal case
            gid_pair = (self.gid[id_1], self.gid[id_2])
            if gid_pair not in self.ged_dict:
                id_1, id_2 = (id_2, id_1)
                gid_pair = (self.gid[id_1], self.gid[id_2])
            real_ged = self.ged[id_1][id_2][0]
            ta_ged = self.ged[id_1][id_2][1:]

            new_data["id_1"] = id_1
            new_data["id_2"] = id_2

            new_data["edge_index_1"] = self.edge_index[id_1]
            new_data["edge_index_2"] = self.edge_index[id_2]
            new_data["features_1"] = self.features[id_1]
            new_data["features_2"] = self.features[id_2]
            new_data["permute"] = [list(range(self.gn[id_1]))] if id_1 == id_2 else self.ged_dict[gid_pair][1]

        elif pair_type == 1:  # delta graphs
            new_data["id"] = id_1
            dg: dict = self.delta_graphs[id_1][id_2]

            real_ged = dg["ta_ged"][0]
            ta_ged = dg["ta_ged"][1:]

            new_data["edge_index_1"] = self.edge_index[id_1]
            new_data["edge_index_2"] = dg["edge_index"]
            new_data["features_1"] = self.features[id_1]
            new_data["features_2"] = dg["features"]
            new_data["permute"] = [dg["permute"]]
        else:
            assert False

        n1, m1 = (self.gn[id_1], self.gm[id_1])
        n2, m2 = (self.gn[id_2], self.gm[id_2]) if pair_type == 0 else (dg["n"], dg["m"])
        new_data["n1"] = n1
        new_data["n2"] = n2
        new_data["ged"] = real_ged

        return new_data

    def score(self, testing_graph_set='test'):
        """
        Scoring on the test set.
        """
        print("\n\nModel evaluation on {} set.\n".format(testing_graph_set))
        if testing_graph_set == 'test':
            testing_graphs = self.testing_graphs
        # elif testing_graph_set == 'test2':
            # testing_graphs = self.testing2_graphs
        elif testing_graph_set == 'val':
            testing_graphs = self.val_graphs
        else:
            assert False

        num = 0  # total testing number
        time_usage = []
        mse = []  # score mse
        mae = []  # ged mae
        num_acc = 0  # the number of exact prediction (pre_ged == gt_ged)
        num_fea = 0  # the number of feasible prediction (pre_ged >= gt_ged)
        rho = []
        tau = []
        pk10 = []
        pk20 = []
        recall = [] # path recall
        precision = [] # path precision
        f1 = [] # path f1 score
        for pair_type, i, j_list in tqdm(testing_graphs, file=sys.stdout):
            pre = []
            gt = []
            t_total = 0
            for j in j_list:
                data = self.pack_graph_pair((pair_type, i, j))
                gt_ged = data["ged"]
                t1 = time.time()
                pre_ged = None
                if self.args.method == "FGWAlign":
                    edge_index_1 = data["edge_index_1"]
                    edge_index_2 = data["edge_index_2"]
                    # remove self-loops
                    edge_index_1 = edge_index_1[:, edge_index_1[0] != edge_index_1[1]]
                    edge_index_2 = edge_index_2[:, edge_index_2[0] != edge_index_2[1]]
                    # construct adjacency matrix
                    g1_adj = torch.sparse_coo_tensor(edge_index_1, torch.ones(edge_index_1.shape[1]), (data["n1"], data["n1"])).to_dense()
                    g2_adj = torch.sparse_coo_tensor(edge_index_2, torch.ones(edge_index_2.shape[1]), (data["n2"], data["n2"])).to_dense()
                    if args.dataset == 'AIDS':
                        g1_labels = data["features_1"].argmax(dim=1)
                        g2_labels = data["features_2"].argmax(dim=1)
                    else:
                        g1_labels = g2_labels = None
                    pre_ged, align_mat = FGWAlign(g1_adj, g2_adj, g1_labels, g2_labels, patience=args.patience, device=args.device, sparse=args.sparse, topk=args.topk, light=args.light)
                    # align_mat = align_mat.to_dense()
                    # pre_ged = round(pre_ged)
                    # pre_ged = None
                elif self.args.method in ['ipfp', 'spectral', 'rrwm', 'astar', 'astar-beam', 'starub']:
                    align_mat = cal_ged_pygm(data, self.args)
                else:
                    align_mat = cal_ged_baselines(data, self.args)
                t2 = time.time()
                t_total += t2-t1
                if align_mat is None:
                    align_mat = torch.eye(data["n1"])
                # to dense if sparse
                align_mat = align_mat.to_dense()
                pre_permute = torch.argmax(align_mat, dim=1).tolist()
                if len(set(pre_permute)) != len(pre_permute):
                    pre_permute = optimal_unique_assignment(align_mat)
                pre_edit_edges, pre_relabel_nodes = self.gen_edit_path(data, pre_permute[:data["n1"]])
                pre_ged = len(pre_edit_edges) + len(pre_relabel_nodes)
                num += 1
                pre.append(pre_ged)
                gt.append(gt_ged)
                mse.append(abs(pre_ged - gt_ged) ** 2)
                mae.append(abs(pre_ged - gt_ged))
            
                best_recall = 0.
                best_precision = 0.
                best_f1 = 0.
                # if pre_ged < gt_ged:
                #     print(pre_ged, gt_ged, align_mat)
                #     exit(0)

                for permute in data["permute"]:
                    edit_edges, relabel_nodes = self.gen_edit_path(data, permute)
                    assert len(edit_edges) + len(relabel_nodes) == gt_ged
                    num_overlap = len(pre_edit_edges & edit_edges) + len(pre_relabel_nodes & relabel_nodes)
                    if pre_ged == 0 and gt_ged == 0:
                        best_recall, best_precision, best_f1 = 1, 1, 1
                        break
                    best_precision = max(best_precision, num_overlap / (pre_ged+1e-6))
                    best_recall = max(best_recall, num_overlap / (gt_ged+1e-6))
                    best_f1 = max(best_f1, 2.0 * num_overlap / (gt_ged + pre_ged))
                if pre_ged == gt_ged:
                    num_acc += 1
                if pre_ged >= gt_ged:
                    num_fea += 1
                # print(best_f1, best_recall, best_precision)
            time_usage.append(t_total)
            rho.append(spearmanr(pre, gt)[0])
            tau.append(kendalltau(pre, gt)[0])
            pk10.append(self.cal_pk(10, pre, gt))
            pk20.append(self.cal_pk(20, pre, gt))
            # print("results: ", rho[-1], tau[-1], pk10[-1], pk20[-1])
            recall.append(best_recall)
            precision.append(best_precision)
            f1.append(best_f1)
        
        time_usage = round(np.mean(time_usage), 3)
        rmse = round(np.sqrt(np.mean(mse)), 3)
        mae = round(np.mean(mae), 3)
        acc = round(num_acc / num, 3)
        fea = round(num_fea / num, 3)
        rho = round(np.mean(rho), 3)
        tau = round(np.mean(tau), 3)
        pk10 = round(np.mean(pk10), 3)
        pk20 = round(np.mean(pk20), 3)
        recall = round(np.mean(recall), 3)
        precision = round(np.mean(precision), 3)
        f1 = round(np.mean(f1), 3)
        print('Method: {}, Acc: {:.3f}, Fea: {:.3f}, MAE: {:.3f}, RMSE: {:.3f}, Rho: {:.3f}, Tau: {:.3f}, P@10: {:.3f}, P@20: {:.3f}, Rec: {:.3f}, Pre: {:.3f}, F1: {:.3f}, time: {:.3f}'.format(
            self.args.method, acc, fea, mae, rmse, rho, tau, pk10, pk20, recall, precision, f1, time_usage))
        now = datetime.datetime.now()
        
        os.makedirs('results', exist_ok=True)
        with open('results/ged_{}.txt'.format(self.args.dataset), 'a+') as f:
            print(now.strftime("%Y_%m_%d_%H_%M_%S"), file=f)
            print('Method {}, Patience: {}, Topk: {}, Light: {}, Acc: {:.3f}, Fea: {:.3f},  MAE: {:.3f}, RMSE: {:.3f}, Rho: {:.3f}, Tau: {:.3f}, P@10: {:.3f}, P@20: {:.3f}, Rec: {:.3f}, Pre: {:.3f}, F1: {:.3f}, time: {:.3f}'.format(self.args.method, self.args.patience, self.args.topk, self.args.light, acc, fea, mae, rmse, rho, tau, pk10, pk20, recall, precision, f1, time_usage), file=f)
            # print('{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}'.format(
            #     acc, fea, rmse, mae, rho, tau, pk10, pk20, recall, precision, f1, time_usage), file=f)
            print("======================================\n", file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GED experiments.")
    parser.add_argument("--method", type=str, default="FGWAlign", help='Model name')
    parser.add_argument("--dataset", type=str, default='AIDS', help="Dataset name")
    parser.add_argument("--patience", type=int, default=1, help="Exploration patience T in FGWAlign.")
    parser.add_argument("--topk", type=int, default=10, help="Diverse projection candidates K in FGWAlign.")
    parser.add_argument("--beam", type=int, default=5, help="Beam width for AStar-beam")
    parser.add_argument("--num-testing-graphs", type=int, default=100, help="Number of testing graph pairs for each graph")
    parser.add_argument("--device", type=str, default='cpu', help="Device to run the model")
    parser.add_argument("--sparse", action="store_true", default=False, help="Whether to use sparse matrix")
    parser.add_argument("--light", action="store_true", default=False, help="Whether to use the light version of FGWAlign")
    args = parser.parse_args()
    print(args)
    trainer = GED_Tester(args)
    trainer.score('test')
