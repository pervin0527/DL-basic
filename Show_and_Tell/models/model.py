import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision.models import resnet152, ResNet152_Weights, efficientnet_b1, EfficientNet_B1_Weights


class EncoderCNN(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        backbone = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
        # backbone = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
        modules = list(backbone.children())[:-1] ## 마지막 출력층은 제외
        self.backbone = nn.Sequential(*modules)

        self.linear = nn.Linear(backbone.fc.in_features, embed_dim) ## embedding layer
        self.bn = nn.BatchNorm1d(embed_dim, momentum=0.01)
        
    def forward(self, images):
        with torch.no_grad():
            features = self.backbone(images)

        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))

        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, num_layers, max_seq_length=20):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size) ## output layer
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions) ## embedding된 토큰 문장.

        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1) ## encoder가 임베딩한 벡터와 임베딩된 caption을 cat
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) ## 패딩을 적용해서 차원을 맞춰줌.
        hiddens, _ = self.lstm(packed) ## 다음 hidden state 계산.
        outputs = self.linear(hiddens[0])

        return outputs
    
    def sample(self, features, states=None):
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states) ## hiddens : (batch_size, 1, hidden_dim)
            outputs = self.linear(hiddens.squeeze(1)) ## outputs :  (batch_size, vocab_size)
            _, predicted = outputs.max(1) ## predicted: (batch_size) 확률이 가장 높은 하나를 선정.
            sampled_ids.append(predicted)

            inputs = self.embed(predicted) ## inputs: (batch_size, embed_dim)
            inputs = inputs.unsqueeze(1) ## inputs: (batch_size, 1, embed_dim)
        sampled_ids = torch.stack(sampled_ids, 1) ## sampled_ids: (batch_size, max_seq_length)

        return sampled_ids
    
    def beam_search(self, features, states=None, beam_width=3):
        k = beam_width
        sequences = [[list(), 0.0]]
        inputs = features.unsqueeze(1)

        # Iterate through the max sequence length
        for _ in range(self.max_seq_length):
            all_candidates = list()
            for seq, score in sequences:
                hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_dim)
                outputs = self.linear(hiddens.squeeze(1))  # outputs: (batch_size, vocab_size)
                log_probs = F.log_softmax(outputs, dim=1)  # log probabilities of words
                topk_log_probs, topk_indices = log_probs.topk(k, dim=1)  # top k log probabilities and their indices

                for i in range(k):
                    candidate = [seq + [topk_indices[0][i].item()], score - topk_log_probs[0][i].item()]
                    all_candidates.append(candidate)
            
            # Order all candidates by score
            ordered = sorted(all_candidates, key=lambda tup: tup[1])
            sequences = ordered[:k]  # Select k best

            # Update inputs for next iteration
            inputs = self.embed(torch.tensor([seq[0][-1] for seq, score in sequences]).to(features.device))
            inputs = inputs.unsqueeze(1)
        
        best_sequence = sequences[0][0]
        return best_sequence