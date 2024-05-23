import os
import re
import torch
import pandas as pd

from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

def english_preprocessing(data, col):
    data[col] = data[col].astype(str)
    data[col] = data[col].apply(lambda x: x.lower())
    data[col] = data[col].apply(lambda x: re.sub("[^A-Za-z\s]", "", x))
    data[col] = data[col].apply(lambda x: re.sub("\s+", " ", x))
    data[col] = data[col].apply(lambda x: " ".join([word for word in x.split()]))
    return data

def french_preprocessing(data, col):
    data[col] = data[col].astype(str)
    data[col] = data[col].apply(lambda x: x.lower())
    data[col] = data[col].apply(lambda x: re.sub(r'\d', '', x))
    data[col] = data[col].apply(lambda x: re.sub(r'\s+', ' ', x))
    data[col] = data[col].apply(lambda x: re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,।]", "", x))
    data[col] = data[col].apply(lambda x: x.strip())
    return data

def filterPair(pair, max_length):
    return len(pair[0].split(' ')) < max_length and len(pair[1].split(' ')) < max_length

def filterPairs(pairs, max_length):
    filtered_pairs = [pair for pair in pairs if filterPair(pair, max_length)]
    print(f"Filtered pairs: {len(filtered_pairs)} / {len(pairs)}")
    return filtered_pairs

def split_data(total_data_dir, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, n_rows=500000, max_length=50):
    df = pd.read_csv(total_data_dir, nrows=n_rows)
    df = df.sample(frac=1).reset_index(drop=True)
    
    # 결측치가 없는 행 제거
    df = df.dropna(subset=['en', 'fr'])

    df = df[df['en'].apply(lambda x: isinstance(x, str))]
    df = df[df['fr'].apply(lambda x: isinstance(x, str))]

    df = english_preprocessing(df, 'en')
    df = french_preprocessing(df, 'fr')
    df = df[df.apply(lambda row: filterPair([row['en'], row['fr']], max_length), axis=1)]

    save_dir = total_data_dir.split('/')[:-1]
    save_dir = '/'.join(save_dir) + '/dataset'
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(f'{save_dir}/train.csv') or not os.path.exists(f'{save_dir}/valid.csv'):
        total_size = len(df)
        train_size = int(total_size * train_ratio)
        valid_size = int(total_size * valid_ratio)
        test_size = total_size - train_size - valid_size

        train_data = df.iloc[:train_size]
        valid_data = df.iloc[train_size:train_size + valid_size]
        test_data = df.iloc[train_size + valid_size:]
        
        train_data.to_csv(f'{save_dir}/train.csv', index=False, encoding='utf-8')
        valid_data.to_csv(f'{save_dir}/valid.csv', index=False, encoding='utf-8')
        test_data.to_csv(f'{save_dir}/test.csv', index=False, encoding='utf-8')

def tokenize_and_build_vocab(lang, pairs, pad_token, sos_token, eos_token, unk_token):
    if lang == 'eng':
        tokenizer = get_tokenizer('spacy', language="en_core_web_trf")
    elif lang == 'fra':
        tokenizer = get_tokenizer('spacy', language="fr_dep_news_trf")
    else:
        raise ValueError(f"Unsupported language: {lang}")

    vocab = build_vocab_from_iterator(
        (tokenizer(pair[0]) if lang == 'eng' else tokenizer(pair[1]) for pair in tqdm(pairs, desc=f"Building vocab for {lang}")), min_freq=2, max_tokens=10000)
    
    vocab.insert_token('<pad>', pad_token)
    vocab.insert_token('<sos>', sos_token)
    vocab.insert_token('<eos>', eos_token)
    vocab.insert_token('<unk>', unk_token)
    vocab.set_default_index(vocab['<unk>'])

    return vocab, tokenizer

def build_vocab(data_dir, src_lang='eng', trg_lang='fra', save_dir=None, max_length=50, n_rows=500000, tokens=[0, 1, 2, 3]):
    df = pd.read_csv(data_dir, nrows=n_rows)
    
    # 결측치가 없는 행 제거
    df = df.dropna(subset=['en', 'fr'])

    # 문자열이 아닌 값 필터링
    df = df[df['en'].apply(lambda x: isinstance(x, str))]
    df = df[df['fr'].apply(lambda x: isinstance(x, str))]

    # 영어 및 프랑스어 전처리 적용
    df = english_preprocessing(df, 'en')
    df = french_preprocessing(df, 'fr')

    pairs = [[row['en'], row['fr']] for _, row in tqdm(df.iterrows(), desc="Preparing pairs")]
    pairs = filterPairs(pairs, max_length)

    src_vocab, src_tokenizer = tokenize_and_build_vocab(src_lang, pairs, tokens[0], tokens[1], tokens[2], tokens[3])
    trg_vocab, trg_tokenizer = tokenize_and_build_vocab(trg_lang, pairs, tokens[0], tokens[1], tokens[2], tokens[3])

    if save_dir:
        torch.save(src_vocab, os.path.join(save_dir, f'src_vocab_{src_lang}.pth'))
        torch.save(trg_vocab, os.path.join(save_dir, f'trg_vocab_{trg_lang}.pth'))

    return src_vocab, src_tokenizer, trg_vocab, trg_tokenizer

class TranslationDataset(Dataset):
    def __init__(self, data_dir, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer, src_lang='eng', max_length=50, n_rows=500000):
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.max_length = max_length

        df = pd.read_csv(data_dir, nrows=n_rows)
        
        # 결측치가 없는 행 제거
        df = df.dropna(subset=['en', 'fr'])

        # 문자열이 아닌 값 필터링
        df = df[df['en'].apply(lambda x: isinstance(x, str))]
        df = df[df['fr'].apply(lambda x: isinstance(x, str))]

        # 영어 및 프랑스어 전처리 적용
        df = english_preprocessing(df, 'en')
        df = french_preprocessing(df, 'fr')

        self.pairs = [[row['en'], row['fr']] for _, row in df.iterrows()]
        self.pairs = filterPairs(self.pairs, max_length)
        if src_lang == 'fra':
            self.pairs = [list(reversed(p)) for p in self.pairs]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_sentence, output_sentence = self.pairs[idx]

        input_tokens = self.src_tokenizer(input_sentence)
        output_tokens = self.trg_tokenizer(output_sentence)

        # max_length를 초과하는 경우 슬라이싱
        if len(input_tokens) > self.max_length - 2:  # <sos>와 <eos>를 위한 공간 확보
            input_tokens = input_tokens[:self.max_length - 2]
        if len(output_tokens) > self.max_length - 2:  # <sos>와 <eos>를 위한 공간 확보
            output_tokens = output_tokens[:self.max_length - 2]

        input_tensor = [self.src_vocab['<sos>']] + [self.src_vocab[token] if token in self.src_vocab else self.src_vocab['<unk>'] for token in input_tokens] + [self.src_vocab['<eos>']]
        output_tensor = [self.trg_vocab['<sos>']] + [self.trg_vocab[token] if token in self.trg_vocab else self.trg_vocab['<unk>'] for token in output_tokens] + [self.trg_vocab['<eos>']]

        return torch.tensor(input_tensor, dtype=torch.long), torch.tensor(output_tensor, dtype=torch.long)

def collate_fn(batch):
    src_batch, trg_batch = [], []
    for src_sample, trg_sample in batch:
        src_batch.append(src_sample)
        trg_batch.append(trg_sample)

    src_batch = pad_sequence(src_batch, padding_value=0)
    trg_batch = pad_sequence(trg_batch, padding_value=0)

    return src_batch, trg_batch
