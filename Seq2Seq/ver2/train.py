import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import math
import yaml
import torch

from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchtext.data.metrics import bleu_score
from torch.utils.tensorboard import SummaryWriter
from nltk.translate.bleu_score import corpus_bleu
from torchtext.datasets import multi30k, Multi30k

from dataset import build_vocab, collate_fn
from model import Encoder, Decoder, Attention, Seq2Seq

multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
multi30k.URL["test"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/test_2016_flickr.de.gz"

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def train(model, optimizer, criterion, cfg, vocab_de, vocab_en, tokenize_de, tokenize_en, device, writer, epoch):
    model.train()
    dataset = list(Multi30k(split='train', language_pair=(cfg['lang']['src'], cfg['lang']['trg'])))
    dataloader = DataLoader(dataset, batch_size=cfg['hyps']['batch_size'], collate_fn=lambda batch: collate_fn(batch, vocab_de, vocab_en, tokenize_de, tokenize_en))

    epoch_loss = 0
    for src, trg in tqdm(dataloader, desc='Train', leave=False):
        src = src.to(device)
        trg = trg.to(device)
        
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['hyps']['grad_clip'])
        optimizer.step()
        
        epoch_loss += loss.item()
    
    epoch_loss /= len(dataloader)
    perplexity = math.exp(epoch_loss)
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    writer.add_scalar('Perplexity/train', perplexity, epoch)
    
    return epoch_loss


def eval(model, criterion, cfg, vocab_de, vocab_en, tokenize_de, tokenize_en, device, writer, epoch):
    model.eval()
    dataset = list(Multi30k(split='valid', language_pair=(cfg['lang']['src'], cfg['lang']['trg'])))
    dataloader = DataLoader(dataset, batch_size=cfg['hyps']['batch_size'], collate_fn=lambda batch: collate_fn(batch, vocab_de, vocab_en, tokenize_de, tokenize_en))
    
    epoch_loss = 0
    references = []
    hypotheses = []

    with torch.no_grad():
        for src, trg in tqdm(dataloader, desc='Valid', leave=False):
            src = src.to(device)
            trg = trg.to(device)

            output = model(src, trg)
            output_dim = output.shape[-1]

            # Keep the original trg tensor for BLEU score calculation
            original_trg = trg[:, 1:]

            pred = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(pred, trg)
            epoch_loss += loss.item()

            # BLEU score calculation
            output_sentences = output.argmax(2).transpose(0, 1).tolist()
            original_trg_sentences = original_trg.tolist()
            
            for ref, hyp in zip(original_trg_sentences, output_sentences):
                ref_sentence = [vocab_en.lookup_token(tok) for tok in ref if tok not in [vocab_en['<pad>'], vocab_en['<sos>'], vocab_en['<eos>']]]
                hyp_sentence = [vocab_en.lookup_token(tok) for tok in hyp if tok not in [vocab_en['<pad>'], vocab_en['<sos>'], vocab_en['<eos>']]]
                references.append([ref_sentence])
                hypotheses.append(hyp_sentence)
    
    epoch_loss /= len(dataloader)
    perplexity = math.exp(epoch_loss)
    writer.add_scalar('Loss/valid', epoch_loss, epoch)
    writer.add_scalar('Perplexity/valid', perplexity, epoch)
    
    bleu = bleu_score(hypotheses, references) * 100
    writer.add_scalar('BLEU/valid', bleu, epoch)
    
    return epoch_loss, bleu


def main():
    with open('./config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg['paths']['save_dir'], exist_ok=True)
    writer = SummaryWriter(log_dir=cfg['paths']['save_dir'])

    special_tokens = ["<unk>", "<pad>", "<sos>", "<eos>"]
    vocab_de, vocab_en, tokenize_de, tokenize_en = build_vocab(special_tokens)
    
    if cfg['lang']['src'] == 'de':
        input_dim, output_dim = len(vocab_de), len(vocab_en)
    else:
        input_dim, output_dim = len(vocab_en), len(vocab_de)

    attention = Attention(cfg['model']['encoder_hidden_dim'], cfg['model']['decoder_hidden_dim'])
    encoder = Encoder(input_dim, cfg['model']['embed_dim'], cfg['model']['encoder_hidden_dim'], cfg['model']['decoder_hidden_dim'], cfg['model']['encoder_drop_prob'])
    decoder = Decoder(output_dim, cfg['model']['embed_dim'], cfg['model']['encoder_hidden_dim'], cfg['model']['decoder_hidden_dim'], cfg['model']['decoder_drop_prob'], attention)
    model = Seq2Seq(encoder, decoder, device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['hyps']['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=cfg['tokens']['pad_token'])

    best_valid_perplexity = float('inf')
    for epoch in range(1, cfg['hyps']['epochs']+1):
        print(f"\nEpoch : {epoch}")
        train_loss = train(model, optimizer, criterion, cfg, vocab_de, vocab_en, tokenize_de, tokenize_en, device, writer, epoch)
        print(f"Train Loss : {train_loss:.4f}, Perplexity : {math.exp(train_loss):.4f}")

        valid_loss, bleu_score = eval(model, criterion, cfg, vocab_de, vocab_en, tokenize_de, tokenize_en, device, writer, epoch)
        valid_perplexity = math.exp(valid_loss)
        print(f"Valid Loss : {valid_loss:.4f}, Perplexity : {valid_perplexity:.4f}, BLEU Score: {bleu_score:.2f}")

        if valid_perplexity < best_valid_perplexity:
            best_valid_perplexity = valid_perplexity
            torch.save(model.state_dict(), os.path.join(cfg['paths']['save_dir'], 'best.pth'))
            print("Model saved as best.pth")

    torch.save(model.state_dict(), os.path.join(cfg['paths']['save_dir'], 'last.pth'))
    print("Model saved as last.pth")

    writer.close()

if __name__ == "__main__":
    main()
