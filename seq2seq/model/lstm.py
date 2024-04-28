import torch
from torch import nn
from torch.nn import functional as F

class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.x2h = nn.Linear(input_dim, 4 * hidden_dim, bias=bias) ## input과 4개의 gate에 대한 weight matrix를 하나로 정의함.
        self.h2h = nn.Linear(hidden_dim, 4 * hidden_dim, bias=bias) ## hidden state
        self.c2c = torch.Tensor(hidden_dim * 3)

    def forward(self, x, hidden):
        hidden_state, cell_state = hidden
        
        gates = self.x2h(x) + self.h2h(hidden_state)
        ci, cf, co = self.c2c.chunk(3, 0)
        
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        
        ingate = torch.sigmoid(ingate + ci * cell_state)
        forgetgate = torch.sigmoid(forgetgate + cf * cell_state)
        cellgate = forgetgate * cell_state + ingate * torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate + co * cellgate)
        
        hm = outgate * F.tanh(cellgate)

        return hm, cellgate
    

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, bias=True):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.layers = nn.ModuleList([LSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim, bias=bias) for i in range(num_layers)])

    def forward(self, x, hidden_states):
        new_hidden_states = []
        new_cell_states = []
        
        for i in range(self.num_layers):
            hidden = hidden_states[i]
            x, cell_state = self.layers[i](x, hidden)
            
            new_hidden_states.append(x)
            new_cell_states.append(cell_state)
        
        return x, (new_hidden_states, new_cell_states)

    def init_hidden(self, batch_size):
        return ([(torch.zeros(batch_size, self.hidden_dim), torch.zeros(batch_size, self.hidden_dim)) for _ in range(self.num_layers)])