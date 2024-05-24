import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import re
import torch
import pickle
import random
import unicodedata

from torch import nn
from torchtext import transforms
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator


def split_data(total_path, train_path, valid_path, test_path, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    with open(total_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    random.shuffle(lines)

    total_size = len(lines)
    train_size = int(total_size * train_ratio)
    valid_size = int(total_size * valid_ratio)
    test_size = total_size - train_size - valid_size

    train_data = lines[:train_size]
    valid_data = lines[train_size:train_size + valid_size]
    test_data = lines[train_size + valid_size:]

    with open(train_path, 'w', encoding='utf-8') as train_file:
        train_file.writelines(train_data)

    with open(valid_path, 'w', encoding='utf-8') as valid_file:
        valid_file.writelines(valid_data)

    with open(test_path, 'w', encoding='utf-8') as test_file:
        test_file.writelines(test_data)


"""
유니코드 문자열을 아스키 문자열로 변환. 이 과정을 통해 텍스트 데이터의 일관성을 높인다.
"""
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

"""
텍스트를 소문자로 변환하고, 불필요한 공백이나 문자가 아닌 문자를 제거해 모델의 학습 데이터 품질을 향상시킨다.
"""
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def filterPair(p, max_length):
    return len(p[0].split(' ')) < max_length and len(p[1].split(' ')) < max_length

def filterPairs(pairs, max_length):
    return [pair for pair in pairs if filterPair(pair, max_length)]


def tokenize_and_build_vocab(lang, pairs, tokens):
    if lang == 'eng':
        tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    elif lang == 'fra':
        tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')
    else:
        raise ValueError(f"Unsupported language: {lang}")

    vocab = build_vocab_from_iterator(tokenizer(pair[0]) if lang == 'eng' else tokenizer(pair[1]) for pair in pairs)
    vocab.insert_token('<pad>', tokens['pad'])
    vocab.insert_token('<sos>', tokens['sos'])
    vocab.insert_token('<eos>', tokens['eos'])
    vocab.insert_token('<unk>', tokens['unk'])
    vocab.set_default_index(vocab['<unk>'])

    return vocab, tokenizer


def build_vocab(data_dir, src_lang='eng', trg_lang='fra', save_dir=None, max_length=50, tokens={'pad' : 0, 'sos' : 1, 'eos' : 2, 'unk' : 3}):
    lines = open(data_dir, encoding='utf-8').read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    pairs = filterPairs(pairs, max_length)

    src_vocab, src_tokenizer = tokenize_and_build_vocab(src_lang, pairs, tokens)
    trg_vocab, trg_tokenizer = tokenize_and_build_vocab(trg_lang, pairs, tokens)

    if save_dir:
        torch.save(src_vocab, os.path.join(save_dir, f'src_vocab_{src_lang}.pth'))
        torch.save(trg_vocab, os.path.join(save_dir, f'trg_vocab_{trg_lang}.pth'))

    return src_vocab, src_tokenizer, trg_vocab, trg_tokenizer

class TranslationDataset(Dataset):
    def __init__(self, data_dir, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer, max_length=50, src_lang='eng'):
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        
        lines = open(data_dir, encoding='utf-8').read().strip().split('\n')
        self.pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
        self.pairs = filterPairs(self.pairs, max_length)
        
        if src_lang == 'fra':
            self.pairs = [list(reversed(p)) for p in self.pairs]
            
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        input_sentence, output_sentence = self.pairs[idx]
        
        input_tokens = self.src_tokenizer(input_sentence)
        output_tokens = self.trg_tokenizer(output_sentence)
        
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


def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        print(f"{path} folder maded")
    else:
        print(f"{path} is already exist.")

def load_pickle(fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data


def save_pickle(data, fname):
    with open(fname, "wb") as f:
        pickle.dump(data, f)


def make_cache(data_path):
    cache_path = f"{data_path}/cache"
    make_dir(cache_path)

    if not os.path.exists(f"{cache_path}/train.pkl"):
        for name in ["train", "val", "test"]:
            pkl_file_name = f"{cache_path}/{name}.pkl"

            with open(f"{data_path}/{name}.en", "r") as file:
                en = [text.rstrip() for text in file]
            
            with open(f"{data_path}/{name}.de", "r") as file:
                de = [text.rstrip() for text in file]
            
            data = [(en_text, de_text) for en_text, de_text in zip(en, de)]
            save_pickle(data, pkl_file_name)

class Multi30kDataset:
    PAD, PAD_IDX = "<pad>", 0
    SOS, SOS_IDX = "<sos>", 1
    EOS, EOS_IDX = "<eos>", 2
    UNK, UNK_IDX = "<unk>", 3
    SPECIALS = {UNK : UNK_IDX, PAD : PAD_IDX, SOS : SOS_IDX, EOS : EOS_IDX}
    print(SPECIALS)

    URL = "https://github.com/multi30k/dataset/raw/master/data/task1/raw"
    FILES = ["test_2016_flickr.de.gz",
             "test_2016_flickr.en.gz",
             "train.de.gz",
             "train.en.gz",
             "val.de.gz",
             "val.en.gz"]
    

    def __init__(self, data_dir, source_language="en", target_language="de", max_seq_len=256, vocab_min_freq=2):
        self.data_dir = data_dir

        self.max_seq_len = max_seq_len
        self.vocab_min_freq = vocab_min_freq
        self.source_language = source_language
        self.target_language = target_language

        ## 데이터 파일 로드.
        self.train = load_pickle(f"{data_dir}/cache/train.pkl")
        self.valid = load_pickle(f"{data_dir}/cache/val.pkl")
        self.test = load_pickle(f"{data_dir}/cache/test.pkl")

        ## tokenizer 정의.
        if self.source_language == "en":
            self.source_tokenizer = get_tokenizer("spacy", "en_core_web_sm")
            self.target_tokenizer = get_tokenizer("spacy", "de_core_news_sm")
        else:
            self.source_tokenizer = get_tokenizer("spacy", "de_core_news_sm")
            self.target_tokenizer = get_tokenizer("spacy", "en_core_web_sm")

        self.src_vocab, self.trg_vocab = self.get_vocab(self.train)
        self.src_transform = self.get_transform(self.src_vocab)
        self.trg_transform = self.get_transform(self.trg_vocab)


    def yield_tokens(self, train_dataset, is_src):
        for text_pair in train_dataset:
            if is_src:
                yield [str(token) for token in self.source_tokenizer(text_pair[0])]
            else:
                yield [str(token) for token in self.target_tokenizer(text_pair[1])]


    def get_vocab(self, train_dataset):
        src_vocab_pickle = f"{self.data_dir}/cache/vocab_{self.source_language}.pkl"
        trg_vocab_pickle = f"{self.data_dir}/cache/vocab_{self.target_language}.pkl"

        if os.path.exists(src_vocab_pickle) and os.path.exists(trg_vocab_pickle):
            src_vocab = load_pickle(src_vocab_pickle)
            trg_vocab = load_pickle(trg_vocab_pickle)
        else:
            src_vocab = build_vocab_from_iterator(self.yield_tokens(train_dataset, True), min_freq=self.vocab_min_freq, specials=self.SPECIALS.keys())
            src_vocab.set_default_index(self.UNK_IDX)

            trg_vocab = build_vocab_from_iterator(self.yield_tokens(train_dataset, False), min_freq=self.vocab_min_freq, specials=self.SPECIALS.keys())
            trg_vocab.set_default_index(self.UNK_IDX)
            
        return src_vocab, trg_vocab
    

    def get_transform(self, vocab):
        return transforms.Sequential(transforms.VocabTransform(vocab),
                                     transforms.Truncate(self.max_seq_len-2),
                                     transforms.AddToken(token=self.SOS_IDX, begin=True),
                                     transforms.AddToken(token=self.EOS_IDX, begin=False),
                                     transforms.ToTensor(padding_value=self.PAD_IDX))


    def collate_fn(self, pairs):
        src = [self.source_tokenizer(pair[0]) for pair in pairs]
        trg = [self.target_tokenizer(pair[1]) for pair in pairs]

        batch_src = self.src_transform(src)
        batch_trg = self.trg_transform(trg)

        batch_size = batch_src.size(0)
        batch_src = batch_src.view(-1, batch_size)
        batch_trg = batch_trg.view(-1, batch_size)

        return (batch_src, batch_trg)
    

    def get_iter(self, batch_size, num_workers):
        train_iter = DataLoader(self.train, collate_fn=self.collate_fn, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid_iter = DataLoader(self.valid, collate_fn=self.collate_fn, batch_size=batch_size, num_workers=num_workers)
        test_iter = DataLoader(self.test, collate_fn=self.collate_fn, batch_size=batch_size, num_workers=num_workers)

        return train_iter, valid_iter, test_iter
    
    
    def translate(self, model, src_sentence: str, decode_func):
        model.eval()
        src = self.src_transform([self.source_tokenizer(src_sentence)]).view(1, -1)
        num_tokens = src.shape[1]
        trg_tokens = decode_func(model, src, max_len=num_tokens + 5, start_symbol=self.SOS_IDX, end_symbol=self.EOS_IDX).flatten().cpu().numpy()
        trg_sentence = " ".join(self.trg_vocab.lookup_tokens(trg_tokens))

        return trg_sentence