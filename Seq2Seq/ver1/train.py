import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import math
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torch.utils.tensorboard import SummaryWriter

from util import set_seed, calculate_bleu_score
# from model_gru import Encoder, AttentionDecoder, Seq2Seq
from model_lstm import Encoder, AttentionDecoder, Seq2Seq
from dataset import TranslationDataset, split_data, build_vocab, collate_fn, Multi30kDataset, make_cache


def train(model, dataloader, optimizer, criterion, vocab_size, grad_clip, device, epoch, writer, eos_token):
    model.train()
    total_loss = 0
    num_batches = 0
    references = []
    hypotheses = []

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

        output = output.argmax(1).view(trg.size(0), -1).cpu().numpy()
        trg = trg.view(trg.size(0), -1).cpu().numpy()

        for i in range(trg.shape[0]):
            ref = trg[i].tolist()
            hyp = output[i].tolist()
            if eos_token in ref:
                ref = ref[:ref.index(eos_token)]
            if eos_token in hyp:
                hyp = hyp[:hyp.index(eos_token)]
            references.append(ref)
            hypotheses.append(hyp)

    train_loss = total_loss / num_batches
    train_perplexity = math.exp(train_loss)
    train_bleu = calculate_bleu_score(references, hypotheses)

    writer.add_scalar('Train/Loss', train_loss, epoch)
    writer.add_scalar('Train/Perplexity', train_perplexity, epoch)
    writer.add_scalar('Train/BLEU', train_bleu, epoch)

    return train_loss, train_perplexity, train_bleu


def valid(model, dataloader, criterion, vocab_size, device, epoch, writer, eos_token):
    model.eval()
    total_loss = 0
    num_batches = 0
    references = []
    hypotheses = []
    decoded_batch_list = []

    with torch.no_grad():
        for src, trg in tqdm(dataloader, desc='Valid', leave=False):
            src = src.to(device)
            trg = trg.to(device)

            output = model(src, trg)
            output = output[1:].view(-1, vocab_size)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            total_loss += loss.item()
            num_batches += 1

            output = output.argmax(1).view(trg.size(0), -1).cpu().numpy()
            trg = trg.view(trg.size(0), -1).cpu().numpy()

            for i in range(trg.shape[0]):
                ref = trg[i].tolist()
                hyp = output[i].tolist()
                if eos_token in ref:
                    ref = ref[:ref.index(eos_token)]
                if eos_token in hyp:
                    hyp = hyp[:hyp.index(eos_token)]
                references.append(ref)
                hypotheses.append(hyp)

            # decoded_batch = model.decode(src, trg, method='beam-search')
            # decoded_batch_list.append(decoded_batch)
    
    # for sentence_index in decoded_batch_list[0]:
    #     decode_text_arr = [trg_vocab.get_itos()[i] for i in sentence_index[0]]
    #     decode_sentence = " ".join(decode_text_arr[1:-1])
    #     print(f"Pred target : {decode_sentence}")

    valid_loss = total_loss / num_batches
    valid_perplexity = math.exp(valid_loss)
    valid_bleu = calculate_bleu_score(references, hypotheses)

    writer.add_scalar('Valid/Loss', valid_loss, epoch)
    writer.add_scalar('Valid/Perplexity', valid_perplexity, epoch)
    writer.add_scalar('Valid/BLEU', valid_bleu, epoch)

    return valid_loss, valid_perplexity, valid_bleu


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = './runs'
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)

    data_dir = '/home/pervinco/Desktop/en-fr/data'
    total_data_dir = f'{data_dir}/eng-fra.txt'
    train_data_dir = f'{data_dir}/train.txt'
    valid_data_dir = f'{data_dir}/valid.txt'
    test_data_dir = f'{data_dir}/test.txt'
    src_lang, trg_lang = 'eng', 'fra'

    epochs = 100
    batch_size = 256
    learning_rate = 0.0001
    weight_decay = 0.000005
    grad_clip = 10.0
    max_length = 50
    hidden_dim = 512
    embed_dim = 512
    encoder_layers = 2
    decoder_layers = 1
    encoder_dropout = 0.5
    decoder_dropout = 0.0
    pad_token = 0
    sos_token = 1
    eos_token = 2
    unk_token = 3
    specials = {'pad' : pad_token, 'sos' : sos_token, 'eos' : eos_token, 'unk' : unk_token}

    if not os.path.exists(train_data_dir) or not os.path.exists(valid_data_dir):
        split_data(total_data_dir, train_data_dir, valid_data_dir, test_data_dir)

    if not os.path.exists(f'{data_dir}/src_vocab_{src_lang}.pth') and not os.path.exists(f'{data_dir}/trg_vocab_{trg_lang}.pth'):
        src_vocab, src_tokenizer, trg_vocab, trg_tokenizer = build_vocab(total_data_dir, src_lang, trg_lang, data_dir, max_length, tokens=specials)
    else:
        src_vocab = torch.load(f'{data_dir}/src_vocab_{src_lang}.pth')
        trg_vocab = torch.load(f'{data_dir}/trg_vocab_{trg_lang}.pth')

        if src_lang == 'eng':
            src_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
            trg_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')
        elif src_lang == 'fra':
            src_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')
            trg_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

    train_dataset = TranslationDataset(train_data_dir, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer, max_length, src_lang)
    valid_dataset = TranslationDataset(valid_data_dir, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer, max_length, src_lang)
    test_dataset = TranslationDataset(test_data_dir, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer, max_length, src_lang)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)


    # data_dir = '/home/pervinco/Datasets'
    # src_lang, trg_lang = 'de', 'en'
    # make_cache(f"{data_dir}/Multi30k")
    # dataset = Multi30kDataset(data_dir=f"{data_dir}/Multi30k", source_language=src_lang,  target_language=trg_lang,  max_seq_len=max_length, vocab_min_freq=1)
    # train_dataloader, valid_dataloader, test_dataloader = dataset.get_iter(batch_size=batch_size, num_workers=4)
    # src_vocab, trg_vocab = dataset.src_vocab, dataset.trg_vocab


    src_vocab_size, trg_vocab_size = len(src_vocab),len(trg_vocab)
    print(f'SRC vocab size : {src_vocab_size}, TRG vocab size : {trg_vocab_size}')

    encoder = Encoder(src_vocab_size, embed_dim, hidden_dim, n_layers=encoder_layers, dropout=encoder_dropout, pad_token=pad_token).to(device)
    decoder = AttentionDecoder(embed_dim, hidden_dim, trg_vocab_size, n_layers=decoder_layers, dropout=decoder_dropout, pad_token=pad_token).to(device)
    seq2seq = Seq2Seq(encoder, decoder, sos_token, eos_token, max_length, device).to(device)
    optimizer = torch.optim.Adam(seq2seq.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token)

    best_val_loss = float('inf')
    for epoch in range(1, epochs+1):
        print(f'\nEpoch : {epoch}')
        train_loss, train_perplexity, train_bleu = train(seq2seq, train_dataloader, optimizer, criterion, trg_vocab_size, grad_clip, device, epoch, writer, eos_token)
        valid_loss, valid_perplexity, valid_bleu = valid(seq2seq, valid_dataloader, criterion, trg_vocab_size, device, epoch, writer, eos_token)
        print(f'Train Loss : {train_loss:.4f}, Train Perplexity : {train_perplexity:.4f}, Train BLEU : {train_bleu:.4f}')
        print(f'Valid Loss : {valid_loss:.4f}, Valid Perplexity : {valid_perplexity:.4f}, Valid BLEU : {valid_bleu:.4f}')
        
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(seq2seq.state_dict(), os.path.join(save_dir, 'best.pth'))
    
    torch.save(seq2seq.state_dict(), os.path.join(save_dir, 'last.pth'))
    test_loss, test_perplexity, test_bleu = valid(seq2seq, test_dataloader, criterion, trg_vocab, device, epoch, writer, eos_token)
    print(f'Test Loss : {test_loss:.4f}, Test Perplexity : {test_perplexity:.4f}, Test BLEU : {test_bleu:.4f}')
    writer.close()


if __name__ == "__main__":
    main()