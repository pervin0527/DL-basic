import os

from torch import dropout
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

from model.encoder import Encoder
from model.decoder import Decoder
from model.seq2seq import Seq2Seq, gradient_constraint
from data.dataset import Multi30kDataset


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Train", leave=False):
        optimizer.zero_grad()
        src = batch["en_ids"].to(device) ## [src_len, batch_size]
        trg = batch["de_ids"].to(device) ## [trg_len, batch_size]

        output = model(src, trg) ## [trg_len, batch_size, trg_vocab_size]
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim) ## [trg_len-1 * batch_size, trg_vocab_size], <sos> 토큰 제외.
        trg = trg[1:].view(-1) ## [(trg_len - 1) * batch_size] , <sos> 토큰 제외.

        loss = criterion(output, trg)
        loss.backward()

        gradient_constraint(optimizer)
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    return average_loss


def valid(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", leave=False):
            src = batch["en_ids"].to(device) ## [src_len, batch_size]
            trg = batch["de_ids"].to(device) ## [trg_len, batch_size]

            output = model(src, trg) ## [trg_len, batch_size, trg_vocab_size]
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim) ## [trg_len-1 * batch_size, trg_vocab_size]
            trg = trg[1:].view(-1) ## [(trg_len - 1) * batch_size] , <sos> 토큰 제외.
            
            loss = criterion(output, trg)
            total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    return average_loss


def main():
    dataset = Multi30kDataset(max_length=1000, lower=True, min_freq=2)
    input_dim = len(dataset.en_vocab)
    output_dim = len(dataset.de_vocab)

    train_dataset, valid_dataset, test_dataset = dataset.get_datasets()
    pad_index = dataset.en_vocab['pad']

    collate_fn = dataset.get_collate_fn(pad_index)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn)

    encoder = Encoder(input_dim, embedding_dim, hidden_dim, num_layers, dropout_prob)
    decoder = Decoder(output_dim, embedding_dim, hidden_dim, num_layers, dropout_prob)
    model = Seq2Seq(encoder, decoder, device)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_index)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, device)
        end_time = time.time()

        scheduler.step()

        valid_loss = valid(model, valid_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{int(epochs)}, Train Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}, Time: {end_time - start_time:.2f}s")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 10
    batch_size = 128
    learning_rate = 0.7
    dropout_prob = 0.5

    embedding_dim = 1000
    hidden_dim = 1000
    num_layers = 4

    main()