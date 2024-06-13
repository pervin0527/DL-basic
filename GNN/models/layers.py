import dgl
import math
import dgl.function as func

from dgl.nn.functional import edge_softmax

from torch import nn
from torch.nn import functional as F

class GraphConvolution(nn.Module):
    def __init__(self, hidden_dim, activation=F.relu, drop_prob=0.2):
        super().__init__()
        self.activation = activation
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(drop_prob)
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, graph:dgl.DGLGraph):
        h0 = graph.ndata['h']

        graph.update_all(func.copy_u('h', 'm'), func.sum('m', 'u_'))
        h = self.activation(self.linear(graph.ndata['u_'])) + h0
        h = self.norm(h)

        h = self.dropout(h)
        graph.ndata['h'] = h
        
        return graph


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, bias=False, activation=F.relu):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation

        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.linear2 = nn.Linear(hidden_dim, output_dim, bias=bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        return x

class GraphAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, mlp_bias=False, drop_prob=0.2, activation=F.relu):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dk = hidden_dim // num_heads
        self.prob = drop_prob
        self.activation = activation

        self.mlp = MultiLayerPerceptron(input_dim=hidden_dim, hidden_dim=2*hidden_dim, output_dim=hidden_dim, bias=mlp_bias, activation=activation)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(drop_prob)

        self.w1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w4 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w5 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w6 = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, graph: dgl.DGLGraph):
        h0 = graph.ndata['h']  # graph nodes
        e_ij = graph.edata['e_ij']  # graph edges

        graph.ndata['u'] = self.w1(h0).view(-1, self.num_heads, self.dk)
        graph.ndata['v'] = self.w2(h0).view(-1, self.num_heads, self.dk)
        graph.edata['x_ij'] = self.w3(e_ij).view(-1, self.num_heads, self.dk)

        graph.apply_edges(func.v_add_e('v', 'x_ij', 'm'))
        graph.apply_edges(func.u_mul_e('u', 'm', 'attn'))
        graph.edata['attn'] = edge_softmax(graph, graph.edata['attn'] / math.sqrt(self.dk))

        graph.ndata['k'] = self.w4(h0).view(-1, self.num_heads, self.dk)
        graph.edata['x_ij'] = self.w5(e_ij).view(-1, self.num_heads, self.dk)
        graph.apply_edges(func.v_add_e('k', 'x_ij', 'm'))

        graph.edata['m'] = graph.edata['attn'] * graph.edata['m']
        graph.update_all(func.copy_e('m', 'm'), func.sum('m', 'h'))

        h = self.w6(h0) + graph.ndata['h'].view(-1, self.hidden_dim)
        h = self.norm(h)

        h = h + self.mlp(h)
        h = self.norm(h)
        h = self.dropout(h)

        graph.ndata['h'] = h
        return graph
