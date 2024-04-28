import torch
from torch import nn

def init_lstm_weights(m):
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param.data, -0.08, 0.08)


def gradient_constraint(optimizer, threshold=5):
    with torch.no_grad():
        total_norm = 0
        for param in optimizer.param_groups[0]['params']:
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)  # L2 norm
                total_norm += param_norm ** 2
        total_norm = total_norm ** 0.5
        
        if total_norm > threshold:
            scaling_factor = threshold / (2 * total_norm)
            for param in optimizer.param_groups[0]['params']:
                if param.grad is not None:
                    param.grad.data.mul_(scaling_factor)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        self.apply(init_lstm_weights)

    def forward(self, src, trg):
        """
        src : [src_len, batch_size]
        trg : [trg_len, batch_size]
        """
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        ## encoding
        hidden, cell = self.encoder(src)
        # outputs, (hidden, cell) = self.encoder(src)

        input = trg[0, :] ## <sos> tokens. t=1 일 때 decoder의 입력.
        decoded_result = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device) ## decoder의 문장을 저장하기 위함.
        for idx in range(1, trg_len):
            """
            output : [batch_size, output_dim]
            hidden : [num_layers, batch_size, hidden_dim]
            cell : [num_layers, batch_size, hidden_dim]
            """
            output, hidden, cell = self.decoder(input, hidden, cell)
            decoded_result[idx] = output

            ## Greedy decoding
            top = output.argmax(1)
            input = top ## t=1의 출력을 t=2에서 입력으로 사용함.

        return decoded_result