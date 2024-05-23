import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import math
import torch
import yaml
import random
import operator
import nltk
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from nltk.translate.bleu_score import sentence_bleu

from model import Encoder, AttentionDecoder, Seq2Seq
from dataset_txt import TranslationDataset, split_data, build_vocab, collate_fn

nltk.download('punkt')

def calculate_bleu(decoded_sentences, reference_sentences):
    bleu_scores = []
    for decoded_sentence, reference_sentence in zip(decoded_sentences, reference_sentences):
        reference_tokens = [reference_sentence]
        decoded_tokens = decoded_sentence
        bleu_score = sentence_bleu(reference_tokens, decoded_tokens)
        bleu_scores.append(bleu_score)
    return sum(bleu_scores) / len(bleu_scores)

def test(model, dataloader, criterion, src_vocab, trg_vocab, device):
    model.eval()
    vocab_size = len(trg_vocab)
    total_loss = 0
    num_batches = 0
    decoded_batch_list = []
    reference_sentences = []
    decoded_sentences = []
    with torch.no_grad():
        for src, trg in tqdm(dataloader, desc='Test', leave=False):
            src = src.to(device)
            trg = trg.to(device)
            output = model(src, trg)
            output = output[1:].view(-1, vocab_size)
            loss = criterion(output, trg[1:].contiguous().view(-1))
            total_loss += loss.item()
            num_batches += 1
            decoded_batch = model.decode(src, trg, method='beam-search')
            decoded_batch_list.append(decoded_batch)
    
    test_loss = total_loss / num_batches
    test_perplexity = math.exp(test_loss)

    samples = []
    if decoded_batch_list:
        for i in range(len(decoded_batch_list)):
            for j in range(len(decoded_batch_list[i][0])):
                src_sentence = src[:, j].cpu().numpy()
                trg_sentence = trg[:, j].cpu().numpy()
                decoded_sentence = decoded_batch_list[i][j][0]  # 첫 번째 배치의 첫 번째 문장

                src_text = ' '.join([src_vocab.get_itos()[idx] for idx in src_sentence if idx not in {pad_token, sos_token, eos_token}])
                trg_text = [trg_vocab.get_itos()[idx] for idx in trg_sentence if idx not in {pad_token, sos_token, eos_token}]
                decoded_text = [trg_vocab.get_itos()[idx] for idx in decoded_sentence if isinstance(idx, int) and idx not in {pad_token, sos_token, eos_token}]

                reference_sentences.append(trg_text)
                decoded_sentences.append(decoded_text)

                if i == 0 and j == 0:
                    samples = [src_text, ' '.join(trg_text), ' '.join(decoded_text)]

    bleu_score = calculate_bleu(decoded_sentences, reference_sentences)
    print(f'BLEU Score: {bleu_score:.4f}')

    return test_loss, test_perplexity, samples, bleu_score

if __name__ == "__main__":
    with open('./config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    batch_size = config['training']['batch_size']
    grad_clip = config['training']['grad_clip']
    max_length = config['training']['max_length']

    embed_dim = config['model']['embed_dim']
    hidden_dim = config['model']['hidden_dim']
    encoder_layers = config['model']['encoder_layers']
    encoder_dropout = config['model']['encoder_dropout']
    decoder_dropout = config['model']['decoder_dropout']

    pad_token = config['tokens']['pad_token']
    sos_token = config['tokens']['sos_token']
    eos_token = config['tokens']['eos_token']
    unk_token = config['tokens']['unk_token']
    src_lang, trg_lang = config['lang']['src'], config['lang']['trg']
    n_rows = config['lang']['n_rows']

    save_dir = config['paths']['save_dir']
    data_dir = config['paths']['data_dir']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_data_dir = f'{data_dir}/eng-fra.txt'
    train_data_dir = f'{data_dir}/train.txt'
    valid_data_dir = f'{data_dir}/valid.txt'
    test_data_dir = f'{data_dir}/test.txt'

    if not os.path.exists(train_data_dir) or not os.path.exists(valid_data_dir):
        split_data(total_data_dir, train_data_dir, valid_data_dir, test_data_dir)

    if not os.path.exists(f'{data_dir}/src_vocab_{src_lang}.pth') and not os.path.exists(f'{data_dir}/trg_vocab_{trg_lang}.pth'):
        src_vocab, src_tokenizer, trg_vocab, trg_tokenizer = build_vocab(train_data_dir, src_lang, trg_lang, data_dir)
    else:
        src_vocab = torch.load(f'{data_dir}/src_vocab_{src_lang}.pth')
        trg_vocab = torch.load(f'{data_dir}/trg_vocab_{trg_lang}.pth')

        if src_lang == 'eng':
            src_tokenizer = get_tokenizer('spacy', language='en_core_web_trf')
            trg_tokenizer = get_tokenizer('spacy', language='fr_dep_news_trf')
        elif src_lang == 'fra':
            src_tokenizer = get_tokenizer('spacy', language='fr_dep_news_trf')
            trg_tokenizer = get_tokenizer('spacy', language='en_core_web_trf')

    src_vocab_size = len(src_vocab)
    trg_vocab_size = len(trg_vocab)

    test_dataset = TranslationDataset(test_data_dir, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer, src_lang)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    encoder = Encoder(src_vocab_size, embed_dim, hidden_dim, n_layers=encoder_layers, dropout=encoder_dropout, pad_token=pad_token).to(device)
    decoder = AttentionDecoder(embed_dim, hidden_dim, trg_vocab_size, n_layers=1, pad_token=pad_token).to(device)
    seq2seq = Seq2Seq(encoder, decoder, sos_token, eos_token, max_length, device).to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token)
    
    model_path = os.path.join(save_dir, 'best.pth')
    if os.path.exists(model_path):
        seq2seq.load_state_dict(torch.load(model_path))
        print(f'Model loaded from {model_path}')
    else:
        raise FileNotFoundError(f'{model_path} not found.')

    test_loss, test_perplexity, test_samples, bleu_score = test(seq2seq, test_dataloader, criterion, src_vocab, trg_vocab, device)
    print(f'Test Loss : {test_loss:.4f}, Test Perplexity : {test_perplexity:.4f}')
    
    print(f"SRC : {test_samples[0]}")
    print(f"TRG : {test_samples[1]}")
    print(f"PRED : {test_samples[2]}")
