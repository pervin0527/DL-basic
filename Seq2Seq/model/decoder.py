from torch import nn
from model.lstm import LSTM

class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        # self.lstm = LSTM(output_dim, hidden_dim, num_layers)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        """
        input : [batch_size]
        hidden : [num_layers * n_directions, batch_size, hidden_dim]
        cell : [num_layers * n_directions, batch_size, hidden_dim]
        """
        input = input.unsqueeze(0) ## [1, batch_size] == [sequence_length, batch_size]
        embedded = self.dropout(self.embedding(input)) ## [1, batch_size, embedding_dim]

        """
        output : [seq_len, batch_size, hidden_dim * n_directions]
        hidden : [num_layers * num_directions, batch_size, hidden_dim]
        cell : [num_layers * num_directions, batch_size, hidden_dim]
        """
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.output_layer(output.squeeze(0)) ## [batch_size, output_dim]

        return prediction, hidden, cell
