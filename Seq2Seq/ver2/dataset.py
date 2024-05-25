import torch

from torchtext.data.utils import get_tokenizer
from torchtext.datasets import multi30k, Multi30k
from torchtext.vocab import build_vocab_from_iterator

multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

def yield_tokens(data_iter, tokenizer):
    for data_sample in data_iter:
        yield tokenizer(data_sample[0])
        yield tokenizer(data_sample[1])


def build_vocab(special_tokens=["<unk>", "<pad>", "<sos>", "<eos>"]):
    tokenize_de = get_tokenizer('spacy', language='de_core_news_sm')
    tokenize_en = get_tokenizer('spacy', language='en_core_web_sm')

    train_iter = Multi30k(split='train', language_pair=('de', 'en'))
    vocab_de = build_vocab_from_iterator(yield_tokens(train_iter, tokenize_de), specials=special_tokens)
    vocab_en = build_vocab_from_iterator(yield_tokens(train_iter, tokenize_en), specials=special_tokens)
    
    vocab_de.set_default_index(vocab_de["<unk>"])
    vocab_en.set_default_index(vocab_en["<unk>"])

    return vocab_de, vocab_en, tokenize_de, tokenize_en


def collate_fn(batch, vocab_de, vocab_en, tokenizer_de, tokenizer_en):
    src_batch, trg_batch = [], []
    for (src_item, trg_item) in batch:
        src_tensor = torch.tensor([vocab_de[token] for token in tokenizer_de(src_item)], dtype=torch.long)
        trg_tensor = torch.tensor([vocab_en[token] for token in tokenizer_en(trg_item)], dtype=torch.long)
        
        src_batch.append(torch.cat([torch.tensor([vocab_de['<sos>']]), src_tensor, torch.tensor([vocab_de['<eos>']])], dim=0))
        trg_batch.append(torch.cat([torch.tensor([vocab_en['<sos>']]), trg_tensor, torch.tensor([vocab_en['<eos>']])], dim=0))

    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=vocab_de['<pad>'])
    trg_batch = torch.nn.utils.rnn.pad_sequence(trg_batch, padding_value=vocab_en['<pad>'])

    return src_batch, trg_batch