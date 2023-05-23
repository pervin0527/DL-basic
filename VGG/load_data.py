import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from glob import glob
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, path, set_name):
        self.images = sorted(glob(f"{path}/{set_name}/*/*"))
        self.labels = [file_path.split('/')[-2] for file_path in self.images]
        self.classes = list(set(self.labels))
        print(f"{set_name} samples : {len(self.images)}")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True)
        ])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)

        label = self.classes.index(self.labels[index])
        label = torch.nn.functional.one_hot(torch.tensor(label), len(self.classes)).type(torch.float32)

        return image, label
    
    def get_classes(self):
        return self.classes