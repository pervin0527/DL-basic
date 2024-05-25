import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer

from util import set_seed
from dataset import TranslationDataset, collate_fn
# from model_gru import Encoder, AttentionDecoder, Seq2Seq
from model_lstm import Encoder, AttentionDecoder, Seq2Seq

def translate_sentence(model, src_sentence, src_vocab, trg_vocab, src_tokenizer, device, max_len=50):
    model.eval()
    tokens = [src_vocab['<sos>']] + [src_vocab[token] for token in src_tokenizer(src_sentence)] + [src_vocab['<eos>']]
    src_tensor = torch.LongTensor(tokens).unsqueeze(1).to(device)
    
    with torch.no_grad():
        outputs = model(src_tensor, None, teacher_forcing_ratio=0)  # teacher_forcing_ratio=0 for no teacher forcing
    
    translated_tokens = outputs.argmax(2).squeeze(1).cpu().numpy()
    translated_sentence = [trg_vocab.lookup_token(token) for token in translated_tokens if token not in [trg_vocab['<sos>'], trg_vocab['<eos>'], trg_vocab['<pad>']]]
    
    return ' '.join(translated_sentence)


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = '/home/pervinco/Desktop/en-fr/data'
    model_path = './runs/best.pth'
    test_data_dir = f'{data_dir}/valid.txt'
    src_lang, trg_lang = 'eng', 'fra'

    max_length = 50
    hidden_dim = 1000
    embed_dim = 1000
    encoder_layers = 2
    decoder_layers = 1
    encoder_dropout = 0.5
    decoder_dropout = 0.0
    pad_token = '<pad>'
    sos_token = '<sos>'
    eos_token = '<eos>'
    unk_token = '<unk>'
    specials = [pad_token, sos_token, eos_token, unk_token]

    src_vocab = torch.load(f'{data_dir}/src_vocab_{src_lang}.pth')
    trg_vocab = torch.load(f'{data_dir}/trg_vocab_{trg_lang}.pth')

    if src_lang == 'eng':
        src_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
        trg_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')
    elif src_lang == 'fra':
        src_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')
        trg_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

    test_dataset = TranslationDataset(test_data_dir, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer, max_length, src_lang)
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)

    src_vocab_size, trg_vocab_size = len(src_vocab), len(trg_vocab)
    encoder = Encoder(src_vocab_size, embed_dim, hidden_dim, n_layers=encoder_layers, dropout=encoder_dropout, pad_token=src_vocab[pad_token]).to(device)
    decoder = AttentionDecoder(embed_dim, hidden_dim, trg_vocab_size, n_layers=decoder_layers, dropout=decoder_dropout, pad_token=trg_vocab[pad_token]).to(device)
    seq2seq = Seq2Seq(encoder, decoder, src_vocab[sos_token], src_vocab[eos_token], max_length, device).to(device)
    seq2seq.load_state_dict(torch.load(model_path))
    
    seq2seq.eval()
    with torch.no_grad():
        for src, trg in tqdm(test_dataloader, desc='Test', leave=False):
            src_sentence = ' '.join([src_vocab.lookup_token(int(token)) for token in src.squeeze(1).cpu().numpy() if token not in [src_vocab[pad_token], src_vocab[sos_token], src_vocab[eos_token]]])
            trg_sentence = ' '.join([trg_vocab.lookup_token(int(token)) for token in trg.squeeze(1).cpu().numpy() if token not in [trg_vocab[pad_token], trg_vocab[sos_token], trg_vocab[eos_token]]])
            model_output = translate_sentence(seq2seq, src_sentence, src_vocab, trg_vocab, src_tokenizer, device, max_length)
            
            print(f"Source Sentence: {src_sentence}")
            print(f"Target Sentence: {trg_sentence}")
            print(f"Model Output: {model_output}")
            print()
            break

if __name__ == "__main__":
    main()