"""
Author: Ambroise Odonnat
Purpose: Contains miscellaneous functions
"""

import csv
import numpy as np
import scipy.sparse as sp

import torch

def normalize_adjacency(A):
    """
    Function that normalizes an adjacency matrix
    """
    n = A.shape[0]
    A = A + sp.identity(n)
    degs = A.dot(np.ones(n))
    inv_degs = np.power(degs, -1)
    D = sp.diags(inv_degs)
    A_normalized = D.dot(A)

    return A_normalized

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Function that converts a Scipy sparse matrix to a sparse Torch tensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def split_data(path_labels, adj, features):
    """
    Function that split data into training and test set
    """
    adj_train = list()
    features_train = list()
    y_train = list()
    adj_test = list()
    features_test = list()
    proteins_test = list()
    with open(path_labels, 'r') as f:
        for i, line in enumerate(f):
            t = line.split(',')
            if len(t[1][:-1]) == 0:
                proteins_test.append(t[0])
                adj_test.append(adj[i])
                features_test.append(features[i])
            else:
                adj_train.append(adj[i])
                features_train.append(features[i])
                y_train.append(int(t[1][:-1]))

    return (adj_train, features_train, y_train,
            adj_test, features_test, proteins_test)

def write_submission(path_submission, proteins_test, y_pred_proba):
    """
    Function to write submissions to a csv file
    """
    with open(path_submission, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        lst = list()
        for i in range(18):
            lst.append('class' + str(i))
        lst.insert(0, "name")
        writer.writerow(lst)
        for i, protein in enumerate(proteins_test):
            lst = y_pred_proba[i, :].tolist()
            lst.insert(0, protein)
            writer.writerow(lst)