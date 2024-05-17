import os
import cv2
import torch
import spacy

from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator

class CocoDataset(Dataset):
    def __init__(self, data_dir, ds_type='train', transform=None, vocab=None, vocab_file='vocab.pth'):
        self.data_dir = data_dir
        self.ds_type = ds_type
        self.coco = COCO(f'{self.data_dir}/annotations/captions_{ds_type}2017.json')
        self.ids = list(self.coco.anns.keys())
        self.transform = transform

        self.tokenizer = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])

        # 단어사전 로드 또는 생성
        if vocab is None:
            if os.path.exists(f'{self.data_dir}/{vocab_file}'):
                self.vocab = self.load_vocab(f'{self.data_dir}/{vocab_file}')
            else:
                self.vocab = self.build_vocab()
                self.save_vocab(self.vocab, f'{self.data_dir}/{vocab_file}')
        else:
            self.vocab = vocab

    def tokenize_caption(self, caption):
        doc = self.tokenizer(caption.lower())
        return [token.text for token in doc if not token.is_punct]

    def build_vocab(self):
        def yield_tokens(captions):
            for caption in tqdm(captions, desc="Building Vocab"):
                yield self.tokenize_caption(caption)
    
        captions = (self.coco.anns[ann_id]['caption'] for ann_id in self.ids)
        vocab = build_vocab_from_iterator(yield_tokens(captions), specials=["<unk>", "<pad>", "<bos>", "<eos>"])
        vocab.set_default_index(vocab["<unk>"])
        
        return vocab

    def save_vocab(self, vocab, filepath):
        torch.save(vocab, filepath)
        print(f"Vocabulary saved to {filepath}")

    def load_vocab(self, filepath):
        vocab = torch.load(filepath)
        print(f"Vocabulary loaded from {filepath}")
        return vocab

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        coco = self.coco
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(f"{self.data_dir}/{self.ds_type}2017/{path}").convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        
        tokens = self.tokenize_caption(caption)
        token_indices = [self.vocab['<bos>']] + [self.vocab[token] for token in tokens] + [self.vocab['<eos>']]

        return image, token_indices

def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, 0)
    lengths = [len(cap) for cap in captions]
    captions = [torch.tensor(cap) for cap in captions]
    padded_captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=0)

    return images, padded_captions, lengths
