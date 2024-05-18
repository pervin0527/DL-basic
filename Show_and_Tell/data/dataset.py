import os
import nltk
import torch

from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class CocoDataset(Dataset):
    def __init__(self, root, json, vocab, transform=None):
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys()) ## 전체 파일 리스트
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        coco = self.coco
        vocab = self.vocab

        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        ## 문장 -> 토큰화 -> 정수형
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)

        return image, target


def collate_fn(data):
    ## 문장의 길이를 기준으로 오름차순 정렬.
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    images = torch.stack(images, 0) ## [batch_size, channels, height, width]

    ## 리스트 형태의 캡션들을 텐서 하나로 합치기(데이터 개수, 문장 내 최대 토큰 개수)
    lengths = [len(cap) for cap in captions] ## 각 캡션의 길이를 저장한 리스트
    targets = torch.zeros(len(captions), max(lengths)).long() ## 모든 캡션을 수용할 수 있는 크기의 2D 텐서 targets를 생성
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end] ## 각 캡션을 targets 텐서에 복사. 이 때 각 캡션의 실제 길이만큼 복사되고, 나머지 부분은 0으로 패딩        
    
    return images, targets, lengths