import math
import torch
import random
import operator

from torch import nn
from torch.nn import functional as F
from queue import PriorityQueue

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, enc_hidden_dim, dec_hidden_dim, dropout_ratio):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.rnn = nn.GRU(embed_dim, enc_hidden_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hidden_dim * 2, dec_hidden_dim)
        self.dropout = nn.Dropout(dropout_ratio)
     
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hidden_dim * 2) + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(self, hidden, enc_outputs):
        batch_size = enc_outputs.shape[1]
        src_len = enc_outputs.shape[0]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        enc_outputs = enc_outputs.permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat((hidden, enc_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)
    

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, enc_hidden_dim, dec_hidden_dim, dropout_ratio, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.rnn = nn.GRU((enc_hidden_dim * 2) + embed_dim, dec_hidden_dim)
        self.fc_out = nn.Linear((enc_hidden_dim * 2) + dec_hidden_dim + embed_dim, output_dim)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, input, hidden, enc_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))

        attention = self.attention(hidden, enc_outputs)
        attention = attention.unsqueeze(1)
        enc_outputs = enc_outputs.permute(1, 0, 2)
        weighted = torch.bmm(attention, enc_outputs)

        weighted = weighted.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        
        return prediction, hidden.squeeze(0)
    

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        enc_outputs, hidden = self.encoder(src)

        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        input = trg[0, :]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, enc_outputs)

            outputs[t] = output
            top1 = output.argmax(1)

            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[t] if teacher_force else top1

        return outputs