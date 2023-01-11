import torch
from torch.utils.data import Dataset
import dgl

class DGLGraphDataset(Dataset):
    def __init__(self, adj, features, edge_features, labels=None, train=True):
        self.adj = adj
        self.features = features
        self.edge_features = edge_features
        self.labels = labels
        self.train = train

    def __len__(self):
        return len(self.adj)

    def __getitem__(self, idx):
        g = dgl.from_scipy(self.adj[idx])
        g.ndata['feat'] = torch.FloatTensor(self.features[idx])
        #g.edata['feat'] = torch.FloatTensor(self.edge_features[idx])
        
        if self.train :
            return g, self.labels[idx]
        else : 
            return g


class DGLGraphDataset_Multimodal(Dataset):
    def __init__(self, adj, features, edge_features, protein_embeddings, labels=None, train=True):
        self.adj = adj
        self.features = features
        self.edge_features = edge_features
        self.labels = labels
        self.train = train
        self.protein_embeddings = protein_embeddings

    def __len__(self):
        return len(self.adj)

    def __getitem__(self, idx):
        g = dgl.from_scipy(self.adj[idx])
        g.ndata['feat'] = torch.FloatTensor(self.features[idx])
        #g.edata['feat'] = torch.FloatTensor(self.edge_features[idx])
        
        if self.train :
            return g, torch.FloatTensor(self.protein_embeddings[idx]), self.labels[idx]
        else : 
            return g, torch.FloatTensor(self.protein_embeddings[idx])