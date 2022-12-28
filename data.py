"""
Author: Ambroise Odonnat
Purpose: Create graphs and features
"""

import numpy as np
import scipy.sparse as sp
import os

def load_data(path=""):
    """
    Function that loads graphs
    """
    graph_indicator = np.loadtxt(os.path.join(path, "graph_indicator.txt"), dtype=np.int64)
    _, graph_size = np.unique(graph_indicator, return_counts=True)

    edges = np.loadtxt(os.path.join(path, "edgelist.txt"), dtype=np.int64, delimiter=",")
    A = sp.csr_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                      shape=(graph_indicator.size, graph_indicator.size))
    A += A.T

    x = np.loadtxt(os.path.join(path, "node_attributes.txt"), delimiter=",")
    edge_attr = np.loadtxt(os.path.join(path, "edge_attributes.txt"), delimiter=",")

    adj = []
    features = []
    edge_features = []
    idx_n = 0
    idx_m = 0
    for i in range(graph_size.size):
        adj.append(A[idx_n:idx_n + graph_size[i], idx_n:idx_n + graph_size[i]])
        edge_features.append(edge_attr[idx_m:idx_m + adj[i].nnz, :])
        features.append(x[idx_n:idx_n + graph_size[i], :])
        idx_n += graph_size[i]
        idx_m += adj[i].nnz

    return adj, features, edge_features


def load_sequences(path = ''):
    sequences = list()
    with open(os.path.join(path, 'sequences.txt'), 'r') as f:
        for line in f:
            sequences.append(line[:-1])

    # Split data into training and test sets
    sequences_train = list()
    sequences_test = list()
    proteins_test = list()
    y_train = list()
    with open(os.path.join(path, 'graph_labels.txt'), 'r') as f:
        for i, line in enumerate(f):
            t = line.split(',')
            if len(t[1][:-1]) == 0:
                proteins_test.append(t[0])
                sequences_test.append(sequences[i])
            else:
                sequences_train.append(sequences[i])
                y_train.append(int(t[1][:-1]))

    return sequences_train, sequences_test

def split_train_test(adj, features, edge_features, path=''):
    # Split data into training and test sets
    adj_train = list()
    features_train = list()
    edge_features_train = list()
    y_train = list()
    adj_test = list()
    features_test = list()
    edge_features_test = list()
    proteins_test = list()
    with open(os.path.join(path, 'graph_labels.txt'), 'r') as f:
        for i,line in enumerate(f):
            t = line.split(',')
            if len(t[1][:-1]) == 0:
                proteins_test.append(t[0])
                adj_test.append(adj[i])
                features_test.append(features[i])
                edge_features_test.append(edge_features[i])

            else:
                adj_train.append(adj[i])
                features_train.append(features[i])
                y_train.append(int(t[1][:-1]))
                edge_features_train.append(edge_features[i])
    return adj_train, features_train, edge_features_train, y_train, adj_test, features_test, edge_features_test, proteins_test
