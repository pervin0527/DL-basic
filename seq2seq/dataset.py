from distutils.command import build
import spacy
import datasets

from torchtext.vocab import build_vocab_from_iterator

class CustomDataset:
    def __init__(self, max_length, min_freq=1, lower=True):
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        self.special_tokens = [self.sos_token, self.eos_token, self.unk_token, self.pad_token]

        self.lower = lower
        self.min_freq = min_freq
        self.max_length = max_length

        ## load dataset
        dataset = datasets.load_dataset("bentrevett/multi30k")
        
        ## {'en': 'Two young, White males are outside near many bushes.', 'de': 'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.'}
        train_data, valid_data, test_data = (dataset["train"], dataset["validation"], dataset["test"])

        ## Tokenizer & Tokenize
        self.en_tokenizer = spacy.load("en_core_web_sm")
        self.de_tokenizer = spacy.load("de_core_news_sm")
        self.train_data, self.valid_data, self.test_data = self.tokenize(train_data, valid_data, test_data)

        ## build Vocab
        self.en_vocab, self.de_vocab = self.build_vocabs()
        assert self.en_vocab[self.unk_token] == self.de_vocab[self.unk_token]
        assert self.en_vocab[self.pad_token] == self.de_vocab[self.pad_token]

        unk_index = self.en_vocab[self.unk_token]
        pad_index = self.en_vocab[self.pad_token]

        self.en_vocab.set_default_index(unk_index)
        self.de_vocab.set_default_index(unk_index)



    def tokenize_process(self, example, src_tokenizer, trg_tokenizer, max_length, lower, sos_token, eos_token):
        en_tokens = [token.text for token in src_tokenizer.tokenizer(example["en"])][:max_length]
        de_tokens = [token.text for token in trg_tokenizer.tokenizer(example["de"])][:max_length]
        
        if lower:
            en_tokens = [token.lower() for token in en_tokens]
            de_tokens = [token.lower() for token in de_tokens]

        en_tokens = [sos_token] + en_tokens + [eos_token]
        de_tokens = [sos_token] + de_tokens + [eos_token]

        return {"en_tokens": en_tokens, "de_tokens": de_tokens}
    

    def tokenize(self, train_set, valid_set, test_set):
        self.train_data = train_set.map(self.tokenize_process, fn_kwargs={"src_tokenizer": self.en_tokenizer,
                                                                          "trg_tokenizer": self.de_tokenizer,
                                                                          "max_length": self.max_length,
                                                                          "lower": self.lower,
                                                                          "sos_token": self.sos_token,
                                                                          "eos_token": self.eos_token})
        
        self.valid_data = valid_set.map(self.tokenize_process, fn_kwargs={"src_tokenizer": self.en_tokenizer,
                                                                          "trg_tokenizer": self.de_tokenizer,
                                                                          "max_length": self.max_length,
                                                                          "lower": self.lower,
                                                                          "sos_token": self.sos_token,
                                                                          "eos_token": self.eos_token})
        
        self.test_data = test_set.map(self.tokenize_process, fn_kwargs={"src_tokenizer": self.en_tokenizer,
                                                                        "trg_tokenizer": self.de_tokenizer,
                                                                        "max_length": self.max_length,
                                                                        "lower": self.lower,
                                                                        "sos_token": self.sos_token,
                                                                        "eos_token": self.eos_token})
        
        return self.train_data, self.valid_data, self.test_data
    

    def build_vocabs(self):
        en_vocab = build_vocab_from_iterator(self.train_data["en_tokens"], min_freq=self.min_freq, specials=self.special_tokens)
        de_vocab = build_vocab_from_iterator(self.train_data["de_tokens"], min_freq=self.min_freq, specials=self.special_tokens)

        return en_vocab, de_vocab