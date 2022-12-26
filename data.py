"""
Author: Ambroise Odonnat
Purpose: Create graphs and features
"""

import numpy as np
import scipy.sparse as sp

def load_data():
    """
    Function that loads graphs
    """
    graph_indicator = np.loadtxt("graph_indicator.txt", dtype=np.int64)
    _, graph_size = np.unique(graph_indicator, return_counts=True)

    edges = np.loadtxt("edgelist.txt", dtype=np.int64, delimiter=",")
    A = sp.csr_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                      shape=(graph_indicator.size, graph_indicator.size))
    A += A.T

    x = np.loadtxt("node_attributes.txt", delimiter=",")
    edge_attr = np.loadtxt("edge_attributes.txt", delimiter=",")

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