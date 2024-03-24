import torch
import random
import numpy as np

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

class VerticalFlip(object):
    def __call__(self, sample):
        img, mask = sample["image"], sample["mask"]

        if random.random() > 0.5:
            img = F.vflip(img)
            mask = F.vflip(mask)

        return {"image" : img, "mask" : mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        
        if random.random() > 0.5:
            img = F.hflip(img)
            mask = F.hflip(mask)

        return {'image': img, 'mask': mask}


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        img_resized = F.resize(img, self.size)
        mask_resized = F.resize(mask, self.size, interpolation=Image.NEAREST)

        return {'image': img_resized, 'mask': mask_resized}


class ToTensor(object):
    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        img_tensor = F.to_tensor(img)
        mask_tensor = F.to_tensor(mask)

        return {'image': img_tensor, 'mask': mask_tensor}
    

class VOCDataset(Dataset):
    def __init__(self, data_dir, image_set="train", transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_dir = f"{self.data_dir}/JPEGImages"
        self.mask_dir = f"{self.data_dir}/SegmentationClass"

        with open(f"{self.data_dir}/ImageSets/Segmentation/{image_set}.txt", "r") as file:
            self.file_names = file.read().splitlines()

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        image_file = f"{self.image_dir}/{self.file_names[idx]}.jpg"
        mask_file = f"{self.mask_dir}/{self.file_names[idx]}.png"

        image = Image.open(image_file).convert("RGB")
        mask = Image.open(mask_file)

        data = {"image" : image, "mask" : mask}
        t_data = self.transform(data)

        return t_data
    
if __name__ == "__main__":
    data_dir = "/home/pervinco/Datasets/PASCAL_VOC/VOCDevkit/VOC2012"
    img_size = (256, 256)

    transform = transforms.Compose([
        Resize((256, 256)),
        ToTensor()
    ])

    dataset = VOCDataset(data_dir, image_set="train", transform=transform)
    sample = dataset[0]
    print(sample["image"].shape, sample["mask"].shape)