import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    """
    Simple message passing model that consists of 2 message passing layers
    and the sum aggregation function
    """

    def __init__(self, input_dim, hidden_dim, dropout, n_class):
        super(GNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, n_class)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_in, adj, idx):
        # first message passing layer
        x = self.fc1(x_in)
        x = self.relu(torch.mm(adj, x))
        x = self.dropout(x)

        # second message passing layer
        x = self.fc2(x)
        x = self.relu(torch.mm(adj, x))

        # sum aggregator
        idx = idx.unsqueeze(1).repeat(1, x.size(1))
        out = torch.zeros(torch.max(idx) + 1, x.size(1)).to(x_in.device)
        out = out.scatter_add_(0, idx, x)

        # batch normalization layer
        out = self.bn(out)

        # mlp to produce output
        out = self.relu(self.fc3(out))
        out = self.dropout(out)
        out = self.fc4(out)

        return F.log_softmax(out, dim=1)


class GATLayer(nn.Module):
    """
    Graph attention layer that consists in an attention embedding layer
    and a message passing layer
    """

    def __init__(self, input_dim, hidden_dim, alpha=0.05):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim, bias=False)
        self.a = nn.Linear(2 * hidden_dim, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x_in, adj):
        z = self.fc(x_in)

        # indices of pairs of nodes connected by an edge
        indices = adj.coalesce().indices()

        # concatenate embeddings of nodes connected by an edge
        h = torch.cat([z[indices[0, :], :], z[indices[1, :], :]], dim=1)
        h = self.a(h)
        h = self.leakyrelu(h)  # leaky relu
        h = torch.exp(h.squeeze())
        unique = torch.unique(indices[0, :])
        t = torch.zeros(unique.size(0), device=x_in.device)
        h_sum = t.scatter_add(0, indices[0, :], h)
        h_norm = torch.gather(h_sum, 0, indices[0, :])
        alpha = torch.div(h, h_norm)

        # attention scores between nodes
        adj_att = torch.sparse.FloatTensor(indices, alpha, torch.Size([x_in.size(0), x_in.size(0)])).to(x_in.device)

        # Message passing
        out = torch.sparse.mm(adj_att, z)

        return out, alpha


class GAT(nn.Module):
    """
    Graph attention model that consists of 2 graph attention layers
    and the sum aggregation function
    """

    def __init__(self, input_dim, hidden_dim, dropout, n_class):
        super(GAT, self).__init__()
        self.mp1 = GATLayer(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_class)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_in, adj, idx):

        # Normalize x_in to avoid inf values in GAT layers
        x_in = (x_in - x_in.mean()) / x_in.std()

        # attention layer
        x, _ = self.mp1(x_in, adj)
        x = self.relu(x)
        x = self.dropout(x)

        # sum aggregator
        idx = idx.unsqueeze(1).repeat(1, x.size(1))
        out = torch.zeros(torch.max(idx) + 1, x.size(1)).to(x_in.device)
        out = out.scatter_add_(0, idx, x)

        # batch normalization layer
        out = self.bn(out)

        # mlp to produce output
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return F.log_softmax(out, dim=1)