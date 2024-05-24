import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import math
import yaml
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer

from model import Seq2Seq, Encoder, AttentionDecoder
from dataset_txt import TranslationDataset, collate_fn

def test(model, dataloader, criterion, src_vocab, trg_vocab, device):
    model.eval()
    vocab_size = len(trg_vocab)
    total_loss = 0
    num_batches = 0
    decoded_batch_list = []

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
            decoded_batch_list.extend(decoded_batch)

    test_loss = total_loss / num_batches
    test_perplexity = math.exp(test_loss)

    # Print source, target, and predicted sentences
    for i, (src_batch, trg_batch, pred_batch) in enumerate(zip(src, trg, decoded_batch_list)):
        for j in range(src_batch.size(0)):
            # src와 trg는 시퀀스 길이 x 배치 사이즈 형태이므로 각각의 시퀀스를 구성
            src_sentence = ' '.join([src_vocab.get_itos()[token] for token in src_batch[:, j].cpu().numpy()])
            trg_sentence = ' '.join([trg_vocab.get_itos()[token] for token in trg_batch[:, j].cpu().numpy()])

            # pred_batch[j]가 리스트일 경우 처리
            if isinstance(pred_batch[j], list):
                pred_sentence = ' '.join([trg_vocab.get_itos()[token] for token in pred_batch[j]])
            else:
                pred_sentence = ' '.join([trg_vocab.get_itos()[token] for token in pred_batch[j][0]])

            print(f"Batch {i+1}, Sentence {j+1}")
            print(f"SRC: {src_sentence}")
            print(f"TRG: {trg_sentence}")
            print(f"PRED: {pred_sentence}")
            print()

    return test_loss, test_perplexity


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open('./config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    batch_size = config['training']['batch_size']
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

    test_data_dir = f'{data_dir}/test.txt'
    src_vocab = torch.load(f'{data_dir}/src_vocab_{src_lang}.pth')
    trg_vocab = torch.load(f'{data_dir}/trg_vocab_{trg_lang}.pth')
    src_vocab_size = len(src_vocab)
    trg_vocab_size = len(trg_vocab)

    if src_lang == 'eng':
        src_tokenizer = get_tokenizer('spacy', language='en_core_web_trf')
        trg_tokenizer = get_tokenizer('spacy', language='fr_dep_news_trf')
    elif src_lang == 'fra':
        src_tokenizer = get_tokenizer('spacy', language='fr_dep_news_trf')
        trg_tokenizer = get_tokenizer('spacy', language='en_core_web_trf')

    test_dataset = TranslationDataset(test_data_dir, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer, src_lang)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    encoder = Encoder(src_vocab_size, embed_dim, hidden_dim, n_layers=encoder_layers, dropout=encoder_dropout, pad_token=pad_token).to(device)
    decoder = AttentionDecoder(embed_dim, hidden_dim, trg_vocab_size, n_layers=1, pad_token=pad_token).to(device)
    seq2seq = Seq2Seq(encoder, decoder, sos_token, eos_token, max_length, device).to(device)
    seq2seq.load_state_dict(torch.load(os.path.join(save_dir, 'best.pth')))

    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token)
    
    test_loss, test_perplexity = test(seq2seq, test_dataloader, criterion, src_vocab, trg_vocab, device)
    print(f'Test Loss : {test_loss:.4f}, Test Perplexity : {test_perplexity:.4f}')