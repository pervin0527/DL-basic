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
                 buildingblock_embedding_dim=64,
                 drop_prob=0.2):
        super().__init__()
        self.readout = readout
        self.num_layers = num_layers

        self.node_embedding = nn.Linear(initial_node_dim, hidden_dim, bias=False)
        self.edge_embedding = nn.Linear(initial_edge_dim, hidden_dim, bias=False)
        self.protein_embedding = nn.Linear(protein_embedding_dim, hidden_dim, bias=False)
        self.buildingblock_embedding = nn.Linear(hidden_dim, buildingblock_embedding_dim, bias=False)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = GraphConvolution(hidden_dim, activation, drop_prob)
            self.layers.append(layer)

        total_embedding_dim = hidden_dim * 2 + buildingblock_embedding_dim * 3
        self.output = nn.Linear(total_embedding_dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, main_graph, buildingblock_graphs, protein_embedding):
        h = self.node_embedding(main_graph.ndata['h'].float())
        e_ij = self.edge_embedding(main_graph.edata['e_ij'].float())

        main_graph.ndata['h'] = h
        main_graph.edata['e_ij'] = e_ij

        for i in range(self.num_layers):
            main_graph = self.layers[i](main_graph)

        main_graph_out = dgl.readout_nodes(main_graph, 'h', op=self.readout)

        buildingblock_outs = []
        for graph in buildingblock_graphs:
            h = self.node_embedding(graph.ndata['h'].float())
            e_ij = self.edge_embedding(graph.edata['e_ij'].float())

            graph.ndata['h'] = h
            graph.edata['e_ij'] = e_ij

            for i in range(self.num_layers):
                graph = self.layers[i](graph)

            buildingblock_out = dgl.readout_nodes(graph, 'h', op=self.readout)
            buildingblock_out = self.buildingblock_embedding(buildingblock_out)
            buildingblock_outs.append(buildingblock_out)

        buildingblock_out = torch.cat(buildingblock_outs, dim=1)

        protein_emb = self.protein_embedding(protein_embedding)

        out = torch.cat((main_graph_out, protein_emb, buildingblock_out), dim=1)
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
                 protein_embedding_dim=1024,
                 buildingblock_embedding_dim=64):
        super().__init__()
        self.readout = readout
        self.num_layers = num_layers

        self.node_embedding = nn.Linear(initial_node_dim, hidden_dim, bias=False)
        self.edge_embedding = nn.Linear(initial_edge_dim, hidden_dim, bias=False)
        self.protein_embedding = nn.Linear(protein_embedding_dim, hidden_dim, bias=False)
        self.buildingblock_embedding = nn.Linear(hidden_dim, buildingblock_embedding_dim, bias=False)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = GraphAttention(hidden_dim, num_heads, mlp_bias, drop_prob, activation)
            self.layers.append(layer)

        total_embedding_dim = hidden_dim * 2 + buildingblock_embedding_dim * 3
        self.output = nn.Linear(total_embedding_dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, main_graph:dgl.DGLGraph, buildingblock_graphs, protein_embedding):
        ## Main graph
        h = self.node_embedding(main_graph.ndata['h'].float())
        e_ij = self.edge_embedding(main_graph.edata['e_ij'].float())

        main_graph.ndata['h'] = h
        main_graph.edata['e_ij'] = e_ij

        for i in range(self.num_layers):
            main_graph = self.layers[i](main_graph)

        main_graph_out = dgl.readout_nodes(main_graph, 'h', op=self.readout)

        ## Building block graphs
        buildingblock_outs = []
        for graph in buildingblock_graphs:
            h = self.node_embedding(graph.ndata['h'].float())
            e_ij = self.edge_embedding(graph.edata['e_ij'].float())

            graph.ndata['h'] = h
            graph.edata['e_ij'] = e_ij

            for i in range(self.num_layers):
                graph = self.layers[i](graph)

            buildingblock_out = dgl.readout_nodes(graph, 'h', op=self.readout)
            buildingblock_out = self.buildingblock_embedding(buildingblock_out)
            buildingblock_outs.append(buildingblock_out)

        buildingblock_out = torch.cat(buildingblock_outs, dim=1)

        ## Target protein
        protein_emb = self.protein_embedding(protein_embedding)

        ## Merge & output
        out = torch.cat((main_graph_out, protein_emb, buildingblock_out), dim=1)
        out = self.output(out)
        out = self.sigmoid(out)

        return out
