from torch import nn
from model.lstm import LSTM

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        # self.lstm = LSTM(input_dim, hidden_dim, num_layers)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        ## src : [src_length, batch_size]
        # batch_size = src.size(1)
        # hidden_states = self.lstm.init_hidden(batch_size)

        # embedded = self.dropout(self.embedding(src))
        # outputs, (hidden, cell) = self.lstm(embedded, hidden_states)

        # return outputs, (hidden, cell)

        ## embedded : [src_length, batch_size, embedding_dim]
        embedded = self.dropout(self.embedding(src))

        """
        outputs : [src_length, batch_size, hidden_dim * n_directions]
        hidden : [num_layers * n_directions, batch_size, hidden_dim] 
        cell : [num_layers * n_directions, batch_size, hidden_dim]
        """
        outputs, (hidden, cell) = self.lstm(embedded)
        
        return hidden, cell