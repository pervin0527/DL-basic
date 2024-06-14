from os import read
import dgl
import torch

from torch import nn
from torch.nn import functional as F

from models.layers import GraphConvolution, GraphAttention

class GCN(nn.Module):
    def __init__(self, 
                 initial_node_dim=29, 
                 initial_edge_dim=6,
                 num_layers=3, 
                 hidden_dim=64,
                 readout='sum',  
                 activation=F.relu, 
                 protein_embedding_dim=1024,
                 drop_prob=0.2):
        super().__init__()
        self.readout = readout
        self.num_layers = num_layers

        self.node_embedding = nn.Linear(initial_node_dim, hidden_dim, bias=False)
        self.edge_embedding = nn.Linear(initial_edge_dim, hidden_dim, bias=False)
        self.protein_embedding = nn.Linear(protein_embedding_dim, hidden_dim, bias=False)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = GraphConvolution(hidden_dim, activation, drop_prob)
            self.layers.append(layer)

        self.output = nn.Linear(hidden_dim * 2, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, graph:dgl.DGLGraph, protein_embedding):
        ## Graph
        h = self.node_embedding(graph.ndata['h'].float())
        e_ij = self.edge_embedding(graph.edata['e_ij'].float())

        graph.ndata['h'] = h
        graph.edata['e_ij'] = e_ij

        for i in range(self.num_layers):
            graph = self.layers[i](graph)

        out = dgl.readout_nodes(graph, 'h', op=self.readout)

        ## Target protein
        protein_emb = self.protein_embedding(protein_embedding)

        ## Merge & output
        out = torch.cat((out, protein_emb), dim=1)
        out = self.output(out)
        out = self.sigmoid(out)

        return out

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
                 protein_embedding_dim=1024):
        super().__init__()
        self.readout = readout
        self.num_layers = num_layers

        self.node_embedding = nn.Linear(initial_node_dim, hidden_dim, bias=False)
        self.edge_embedding = nn.Linear(initial_edge_dim, hidden_dim, bias=False)
        self.protein_embedding = nn.Linear(protein_embedding_dim, hidden_dim, bias=False)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = GraphAttention(hidden_dim, num_heads, mlp_bias, drop_prob, activation)
            self.layers.append(layer)

        self.output = nn.Linear(hidden_dim * 2, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, graph:dgl.DGLGraph, protein_embedding):
        ## Graph
        h = self.node_embedding(graph.ndata['h'].float())
        e_ij = self.edge_embedding(graph.edata['e_ij'].float())

        graph.ndata['h'] = h
        graph.edata['e_ij'] = e_ij

        for i in range(self.num_layers):
            graph = self.layers[i](graph)

        out = dgl.readout_nodes(graph, 'h', op=self.readout)

        ## Target protein
        protein_emb = self.protein_embedding(protein_embedding)

        ## Merge & output
        out = torch.cat((out, protein_emb), dim=1)
        out = self.output(out)
        out = self.sigmoid(out)

        return out