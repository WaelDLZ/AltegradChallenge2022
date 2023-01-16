import torch
import torch.nn
import torch.nn.functional as F
from layers import ConvPoolReadout
from dgl.nn.pytorch.conv import GraphConv, SAGEConv, GATConv

from dgl.nn.pytorch.glob import SumPooling, GlobalAttentionPooling, AvgPooling


class HGPSLModel(torch.nn.Module):
    r"""

    Description
    -----------
    The graph classification model using HGP-SL pooling.

    Parameters
    ----------
    in_feat : int
        The number of input node feature's channels.
    out_feat : int
        The number of output node feature's channels.
    hid_feat : int
        The number of hidden state's channels.
    dropout : float, optional
        The dropout rate. Default: 0
    pool_ratio : float, optional
        The pooling ratio for each pooling layer. Default: 0.5
    conv_layers : int, optional
        The number of graph convolution and pooling layers. Default: 3
    sample : bool, optional
        Whether use k-hop union graph to increase efficiency.
        Currently we only support full graph. Default: :obj:`False`
    sparse : bool, optional
        Use edge sparsemax instead of edge softmax. Default: :obj:`True`
    sl : bool, optional
        Use structure learining module or not. Default: :obj:`True`
    lamb : float, optional
        The lambda parameter as weight of raw adjacency as described in the
        HGP-SL paper. Default: 1.0
    """

    def __init__(
            self,
            in_feat: int,
            out_feat: int,
            hid_feat: int,
            dropout: float = 0.5,
            pool_ratio: float = 0.5,
            conv_layers: int = 3,
            sample: bool = False,
            sparse: bool = True,
            sl: bool = True,
            lamb: float = 1.0,
    ):
        super(HGPSLModel, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.hid_feat = hid_feat
        self.dropout = dropout
        self.num_layers = conv_layers
        self.pool_ratio = pool_ratio

        convpools = []
        for i in range(conv_layers):
            c_in = in_feat if i == 0 else hid_feat
            c_out = hid_feat
            use_pool = i != conv_layers - 1
            convpools.append(
                ConvPoolReadout(
                    c_in,
                    c_out,
                    pool_ratio=pool_ratio,
                    sample=sample,
                    sparse=sparse,
                    sl=sl,
                    lamb=lamb,
                    pool=use_pool,
                )
            )
        self.convpool_layers = torch.nn.ModuleList(convpools)

        self.lin1 = torch.nn.Linear(hid_feat * 2, hid_feat)
        self.lin2 = torch.nn.Linear(hid_feat, hid_feat // 2)
        self.lin3 = torch.nn.Linear(hid_feat // 2, self.out_feat)

    def forward(self, graph, n_feat):
        final_readout = None
        e_feat = None

        for i in range(self.num_layers):
            graph, n_feat, e_feat, readout = self.convpool_layers[i](
                graph, n_feat, e_feat
            )
            if final_readout is None:
                final_readout = readout
            else:
                final_readout = final_readout + readout

        n_feat = F.relu(self.lin1(final_readout))
        n_feat = F.dropout(n_feat, p=self.dropout, training=self.training)
        n_feat = self.lin2(n_feat)
        n_feat = F.relu(n_feat)
        n_feat = F.dropout(n_feat, p=self.dropout, training=self.training)
        n_feat = self.lin3(n_feat)

        return final_readout, F.log_softmax(n_feat, dim=-1)


class MultimodalModel(torch.nn.Module):
    def __init__(
            self,
            graph_model,
            n_classes,
            h_dim,
            dim_protein_embedding,
            dim_graph_embedding,
            dropout: float = 0.2,

    ):
        super(MultimodalModel, self).__init__()
        self.graph_model = graph_model
        self.n_classes = n_classes
        self.dim_protein_embedding = dim_protein_embedding
        self.dim_graph_embedding = dim_graph_embedding

        self.h_dim = h_dim
        self.dropout = dropout

        self.lin0_a = torch.nn.Linear(2 * self.dim_graph_embedding, self.h_dim)
        self.lin0_b = torch.nn.Linear(self.dim_protein_embedding, self.h_dim)

        self.lin1 = torch.nn.Linear(2 * self.h_dim, self.h_dim)
        # self.lin2 = torch.nn.Linear(self.h_dim, self.h_dim // 2)
        self.lin3 = torch.nn.Linear(self.h_dim, self.n_classes)

    def forward(self, graph, n_feat, protein_embedding):
        graph_embedding, _ = self.graph_model(graph, n_feat)
        # graph_embedding = self.lin0_a(graph_embedding)

        protein_embedding = self.lin0_b(protein_embedding)

        cat_embedding = torch.cat([graph_embedding, protein_embedding], dim=1)

        n_feat = F.relu(self.lin1(cat_embedding))
        n_feat = F.dropout(n_feat, p=self.dropout, training=self.training)
        n_feat = self.lin3(n_feat)

        return F.log_softmax(n_feat, dim=-1)


class GNN(torch.nn.Module):
    def __init__(self,
                 in_feat: int,
                 out_feat: int,
                 hid_feat: int,
                 dropout: float = 0.5,
                 graph_layers=None,
                 agg='mean',
                 num_heads=1):
        super(GNN, self).__init__()

        self.mp1 = GraphConv(in_feat, hid_feat)
        self.mp2 = GraphConv(hid_feat, hid_feat)

        self.graph_layers = graph_layers

        if graph_layers == 'SAGE':
            self.mp1 = SAGEConv(in_feat, hid_feat, aggregator_type=agg)
            self.mp2 = SAGEConv(hid_feat, hid_feat, aggregator_type=agg)
        if graph_layers == 'GAT':
            self.mp1 = GATConv(in_feat, hid_feat, num_heads=num_heads)
            self.mp2 = GATConv(hid_feat, hid_feat, num_heads=num_heads)

        self.fc1 = torch.nn.Linear(hid_feat, hid_feat)
        self.fc2 = torch.nn.Linear(hid_feat, out_feat)

        self.bn = torch.nn.BatchNorm1d(hid_feat)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

        self.readout = SumPooling()

    def forward(self, graph, n_feat):
        if self.graph_layers == 'GAT':
            x = self.mp1(graph, n_feat).squeeze(-2)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.mp2(graph, x).squeeze(-2)
            x = self.relu(x)
        else:
            x = self.relu(self.mp1(graph, n_feat))
            x = self.dropout(x)
            x = self.relu(self.mp2(graph, x))

        embedding = self.readout(graph, x)

        # batch normalization layer
        out = self.bn(embedding)

        # mlp to produce output
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return embedding, torch.nn.functional.log_softmax(out, dim=-1)


class GNN_roman(torch.nn.Module):
    def __init__(self,
                 in_feat: int,
                 out_feat: int,
                 hid_feat: int,
                 dropout: float = 0.5,
                 graph_layers=None,
                 agg='mean',
                 num_heads=1):
        super(GNN_roman, self).__init__()

        self.mp1 = GraphConv(in_feat, hid_feat)
        self.mp2 = GraphConv(hid_feat, hid_feat // 2)
        self.mp3 = GraphConv(hid_feat // 2, hid_feat // 2)

        self.graph_layers = graph_layers

        if graph_layers == 'SAGE':
            self.mp1 = SAGEConv(in_feat, hid_feat, aggregator_type=agg)
            self.mp2 = SAGEConv(hid_feat, hid_feat, aggregator_type=agg)
        if graph_layers == 'GAT':
            self.mp1 = GATConv(in_feat, hid_feat, num_heads=num_heads)
            self.mp2 = GATConv(hid_feat, hid_feat, num_heads=num_heads)

        self.fc1 = torch.nn.Linear(hid_feat // 2, hid_feat // 2)
        self.fc2 = torch.nn.Linear(hid_feat // 2, out_feat)

        self.bn = torch.nn.BatchNorm1d(hid_feat // 2)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

        self.readout = SumPooling()

    def forward(self, graph, n_feat):
        if self.graph_layers == 'GAT':
            x = self.mp1(graph, n_feat).squeeze(-2)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.mp2(graph, x).squeeze(-2)
            x = self.relu(x)
        else:
            x = self.relu(self.mp1(graph, n_feat))
            x = self.dropout(x)
            x = self.relu(self.mp2(graph, x))
            x = self.dropout(x)
            x = self.relu(self.mp3(graph, x))

        embedding = self.readout(graph, x)

        # batch normalization layer
        out = self.bn(embedding)

        # mlp to produce output
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return embedding, torch.nn.functional.log_softmax(out, dim=-1)




class GNN_multiple(torch.nn.Module):
    def __init__(self,
                 in_feat: int,
                 out_feat: int,
                 hid_feat: int,
                 dropout: float = 0.5,
                 graph_layers=None,
                 agg='mean',
                 num_heads=1):
        super(GNN_multiple, self).__init__()
        self.gnn1 = GNN(in_feat, out_feat, hid_feat, dropout, graph_layers, agg, num_heads)
        self.gnn2 = GNN(in_feat, out_feat, hid_feat, dropout, graph_layers, agg, num_heads)
        self.gnn3 = GNN(in_feat, out_feat, hid_feat, dropout, graph_layers, agg, num_heads)

        self.bn = torch.nn.BatchNorm1d(3*hid_feat)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

        self.fc1 = torch.nn.Linear(3*hid_feat, hid_feat)
        self.fc2 = torch.nn.Linear(hid_feat, out_feat)

    def forward(self, g1, g2, g3):
        output1, _ = self.gnn1(g1, g1.ndata["feat"])
        output2, _ = self.gnn1(g2, g2.ndata["feat"])
        output3, _ = self.gnn1(g3, g3.ndata["feat"])
        embedding = torch.cat([output1, output2, output3], dim=-1)
        
        out = self.bn(embedding)


        # mlp to produce output
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return embedding, torch.nn.functional.log_softmax(out, dim=-1)
