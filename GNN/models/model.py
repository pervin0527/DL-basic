import dgl
import torch

from torch import nn
from torch.nn import functional as F

from models.layers import GraphAttention

class GAT(nn.Module):
    def __init__(self, 
                 num_layers=5, 
                 hidden_dim=64, 
                 num_heads=4, 
                 drop_prob=0.2, 
                 mlp_bias=False, 
                 readout='sum', 
                 activation=F.relu, 
                 initial_node_dim=29, 
                 initial_edge_dim=6,
                 num_proteins=3):
        super().__init__()
        self.readout = readout
        self.num_layers = num_layers
        self.num_proteins = num_proteins

        self.node_embedding = nn.Linear(initial_node_dim, hidden_dim, bias=False)
        self.edge_embedding = nn.Linear(initial_edge_dim, hidden_dim, bias=False)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = GraphAttention(hidden_dim, num_heads, mlp_bias, drop_prob, activation)
            self.layers.append(layer)

        self.output = nn.Linear(hidden_dim + num_proteins, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, graph:dgl.DGLGraph, protein_idx):
        h = self.node_embedding(graph.ndata['h'].float())
        e_ij = self.edge_embedding(graph.edata['e_ij'].float())

        graph.ndata['h'] = h
        graph.edata['e_ij'] = e_ij

        for i in range(self.num_layers):
            graph = self.layers[i](graph)

        out = dgl.readout_nodes(graph, 'h', op=self.readout)
        protein_onehot = F.one_hot(protein_idx, num_classes=self.num_proteins).float()
        out = torch.cat((out, protein_onehot), dim=1)

        out = self.output(out)
        out = self.sigmoid(out)

        return out
