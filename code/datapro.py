import numpy as np
from scipy.sparse import coo_matrix
import os
import torch
import csv
import torch.utils.data.dataset as Dataset


def dense2sparse(matrix: np.ndarray):
    mat_coo = coo_matrix(matrix)
    edge_idx = np.vstack((mat_coo.row, mat_coo.col))
    return edge_idx, mat_coo.data


def loading_data(param):
    md_matrix = np.loadtxt(os.path.join(r'D:\YD\MFERL-main\datasets\CircBANK\matrix_821-2115.csv'), dtype=int, delimiter=',')

    # get the edge of positives samples
    rng = np.random.default_rng(seed=42)  # 固定训练测试
    pos_samples = np.where(md_matrix == 1)
    pos_samples_shuffled = rng.permutation(pos_samples, axis=1)

    # get the edge of negative samples
    rng = np.random.default_rng(seed=42)
    neg_samples = np.where(md_matrix == 0)
    neg_samples_shuffled = rng.permutation(neg_samples, axis=1)[:, : pos_samples_shuffled.shape[1]]

    edge_idx_dict = dict()

    train_pos_edges = pos_samples_shuffled
    train_neg_edges = neg_samples_shuffled
    train_pos_edges = train_pos_edges.T
    train_neg_edges = train_neg_edges.T
    train_true_label = np.hstack((np.ones(train_pos_edges.shape[0]), np.zeros(train_neg_edges.shape[0])))
    train_true_label = np.array(train_true_label, dtype='float32')
    train_edges = np.vstack((train_pos_edges, train_neg_edges))
    # np.savetxt('./train_test/train_pos.csv', train_pos_edges, delimiter=',')
    # np.savetxt('./train_test/train_neg.csv', train_neg_edges, delimiter=',')

    edge_idx_dict['train_Edges'] = train_edges
    edge_idx_dict['train_Labels'] = train_true_label

    edge_idx_dict['true_md'] = md_matrix  ##*

    return edge_idx_dict


def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return torch.Tensor(md_data)


def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.LongTensor(edge_index)


class CVEdgeDataset(Dataset.Dataset):
    def __init__(self, edges, labels):
        self.Data = edges
        self.Label = labels

    def __len__(self):
        return len(self.Label)

    def __getitem__(self, index):
        data = self.Data[index]
        label = self.Label[index]
        return data, label
