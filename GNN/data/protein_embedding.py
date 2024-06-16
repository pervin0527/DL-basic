import json
import torch
from torch import nn
from transformers import BertModel, BertTokenizer

class FineTuneProteinBertModel(nn.Module):
    def __init__(self):
        super(FineTuneProteinBertModel, self).__init__()
        self.bert = BertModel.from_pretrained('Rostlab/prot_bert')
        self.fc = nn.Linear(self.bert.config.hidden_size, 128)  # 예시로 임베딩 크기를 128로 설정

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # CLS token의 출력
        return self.fc(pooled_output)

def precompute_embeddings(seq_path, output_path, model):
    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert')

    with open(seq_path, 'r') as file:
        protein_seq_dicts = json.load(file)

    embeddings = {}
    for protein_name, sequence in protein_seq_dicts.items():
        inputs = tokenizer(sequence, return_tensors="pt")
        with torch.no_grad():
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
        embeddings[protein_name] = outputs.squeeze().tolist()

    with open(output_path, 'w') as file:
        json.dump(embeddings, file)

if __name__ == "__main__":
    seq_path = '/home/pervinco/Datasets/leash-bio/protein_sequence.json'
    output_path = '/home/pervinco/Datasets/leash-bio/protein_embeddings.json'
    fine_tune_model = FineTuneProteinBertModel()

    precompute_embeddings(seq_path, output_path, fine_tune_model)
