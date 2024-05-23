import os
import pickle

from torchtext import transforms
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator

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
    UNK, UNK_IDX = "<unk>", 0
    PAD, PAD_IDX = "<pad>", 1
    SOS, SOS_IDX = "<sos>", 2
    EOS, EOS_IDX = "<eos>", 3
    SPECIALS = {UNK : UNK_IDX, PAD : PAD_IDX, SOS : SOS_IDX, EOS : EOS_IDX}

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