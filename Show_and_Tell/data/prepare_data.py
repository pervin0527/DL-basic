import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import nltk
import pickle
import argparse

from PIL import Image
Image.ANTIALIAS = Image.LANCZOS

from collections import Counter
from pycocotools.coco import COCO
from torchtext.data.utils import get_tokenizer


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(json, threshold):
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

    words = [word for word, cnt in counter.items() if cnt >= threshold]
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def resize_image(image, size):
    return image.resize(size, Image.ANTIALIAS)


def resize_images(image_dir, output_dir, size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)
    for i, image in enumerate(images):
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                img = resize_image(img, size)
                img.save(os.path.join(output_dir, image), img.format)
        if (i+1) % 100 == 0:
            print ("[{}/{}] Resized the images and saved into '{}'."
                   .format(i+1, num_images, output_dir))


def main(args):
    vocab = build_vocab(json=args.caption_path, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))

    image_dir = args.image_dir
    output_dir = args.output_dir
    image_size = [args.image_size, args.image_size]
    resize_images(image_dir, output_dir, image_size)

    image_dir = args.image_dir.replace('train', 'val')
    output_dir = args.output_dir.replace('train', 'val')
    resize_images(image_dir, output_dir, image_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--caption_path', type=str,  default='/home/pervinco/Datasets/COCO2017/annotations/captions_train2017.json',  help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='/home/pervinco/Datasets/COCO2017/vocab.pkl',  help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=5,  help='minimum word count threshold')
    
    parser.add_argument('--image_dir', type=str, default='/home/pervinco/Datasets/COCO2017/train2017/', help='directory for train images')
    parser.add_argument('--output_dir', type=str, default='/home/pervinco/Datasets/COCO2017/train_resized2017/', help='directory for saving resized images')
    parser.add_argument('--image_size', type=int, default=256, help='size for image after processing')
    
    args = parser.parse_args()
    main(args)
