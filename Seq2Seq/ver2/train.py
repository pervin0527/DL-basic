import os

from regex import P
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import math
import yaml
import torch
import random

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torch.utils.tensorboard import SummaryWriter

from model import Encoder, AttentionDecoder, Seq2Seq
from dataset import Multi30kDataset, make_cache

def train(model, dataloader, optimizer, criterion, vocab_size, grad_clip, device, epoch, writer):
    model.train()
    total_loss = 0
    num_batches = 0
    for src, trg in tqdm(dataloader, desc='Train', leave=False):
        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg)
        output = output[1:].view(-1, vocab_size)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    train_loss = total_loss / num_batches
    train_perplexity = math.exp(train_loss)

    writer.add_scalar('Train/Loss', train_loss, epoch)
    writer.add_scalar('Train/Perplexity', train_perplexity, epoch)

    return train_loss, train_perplexity


def valid(model, dataloader, criterion, trg_vocab, device, epoch, writer):
    model.eval()
    vocab_size = len(trg_vocab)
    total_loss = 0
    num_batches = 0
    decoded_batch_list = []

    with torch.no_grad():
        for src, trg in tqdm(dataloader, desc='Valid', leave=False):
            src = src.to(device)
            trg = trg.to(device)

            output = model(src, trg)
            output = output[1:].view(-1, vocab_size)
            loss = criterion(output, trg[1:].contiguous().view(-1))
            total_loss += loss.item()
            num_batches += 1
            decoded_batch = model.decode(src, trg, method='beam-search')
            decoded_batch_list.append(decoded_batch)
    
    for sentence_index in decoded_batch_list[0]:
        decode_text_arr = [trg_vocab.get_itos()[i] for i in sentence_index[0]]
        decode_sentence = " ".join(decode_text_arr[1:-1])
        print(f"Pred target : {decode_sentence}")

    valid_loss = total_loss / num_batches
    valid_perplexity = math.exp(valid_loss)

    writer.add_scalar('Valid/Loss', valid_loss, epoch)
    writer.add_scalar('Valid/Perplexity', valid_perplexity, epoch)

    return valid_loss, valid_perplexity


if __name__ == "__main__":
    with open('./config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']
    grad_clip = config['training']['grad_clip']
    max_length = config['training']['max_length']

    embed_dim = config['model']['embed_dim']
    hidden_dim = config['model']['hidden_dim']
    encoder_layers = config['model']['encoder_layers']
    decoder_layers = config['model']['decoder_layers']
    encoder_dropout = config['model']['encoder_dropout']
    decoder_dropout = config['model']['decoder_dropout']

    unk_token = config['tokens']['unk_token']
    pad_token = config['tokens']['pad_token']
    sos_token = config['tokens']['sos_token']
    eos_token = config['tokens']['eos_token']
    src_lang, trg_lang = config['lang']['src'], config['lang']['trg']
    print(f'UNK : {unk_token}, PAD : {pad_token}, SOS : {sos_token}, EOS : {eos_token}')

    save_dir = config['paths']['save_dir']
    data_dir = config['paths']['data_dir']

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)

    make_cache(f"{data_dir}/Multi30k")
    dataset = Multi30kDataset(data_dir=f"{data_dir}/Multi30k", source_language=src_lang,  target_language=trg_lang,  max_seq_len=max_length, vocab_min_freq=2)
    train_dataloader, valid_dataloader, test_dataloader = dataset.get_iter(batch_size=batch_size, num_workers=num_workers)
    src_vocab, trg_vocab = dataset.src_vocab, dataset.trg_vocab
    src_vocab_size, trg_vocab_size = len(src_vocab), len(trg_vocab)
    print(src_vocab_size, trg_vocab_size)

    encoder = Encoder(src_vocab_size, embed_dim, hidden_dim, n_layers=encoder_layers, dropout=encoder_dropout, pad_token=pad_token).to(device)
    decoder = AttentionDecoder(embed_dim, hidden_dim, trg_vocab_size, n_layers=1, pad_token=pad_token).to(device)
    seq2seq = Seq2Seq(encoder, decoder, sos_token, eos_token, max_length, device).to(device)
    optimizer = torch.optim.Adam(seq2seq.parameters(), lr=learning_rate)

    # optimizer = torch.optim.Adadelta(seq2seq.parameters(), rho=0.95, eps=1e-6)
    print("ignore_index : ", pad_token)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token)
    
    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        print(f'\nEpoch : {epoch}')
        train_loss, train_perplexity = train(seq2seq, train_dataloader, optimizer, criterion, trg_vocab_size, grad_clip, device, epoch, writer)
        valid_loss, valid_perplexity = valid(seq2seq, valid_dataloader, criterion, trg_vocab, device, epoch, writer)
        print(f'Train Loss : {train_loss:.4f}, Train Perplexity : {train_perplexity:.4f}')
        print(f'Valid Loss : {valid_loss:.4f}, Valid Perplexity : {valid_perplexity:.4f}')
        
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(seq2seq.state_dict(), os.path.join(save_dir, 'best.pth'))
    
    torch.save(seq2seq.state_dict(), os.path.join(save_dir, 'last.pth'))
    test_loss, test_perplexity = valid(seq2seq, test_dataloader, criterion, src_vocab, trg_vocab, device, epoch, writer)
    writer.close()
