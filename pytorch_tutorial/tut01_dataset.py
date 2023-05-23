import os
import cv2
import torch
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, ds_dir, ds_type, transform):
        ds_path = os.path.join(ds_dir, ds_type)
        self.classes = os.listdir(ds_path)
        
        self.file_list = []
        for cls in self.classes:
            files = os.listdir(os.path.join(ds_dir, ds_type, cls))
            files = [f"{ds_path}/{cls}/{file}" for file in files]
            self.file_list.extend(files)

        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        label = file_path.split('/')[-2]
        label = torch.tensor(self.classes.index(label))
        label = torch.zeros(len(self.classes), dtype=torch.float).scatter_(dim=0, index=label, value=1)

        image = cv2.imread(self.file_list[idx])
        image = self.transform(image)

        return image, label

data_dir = "/home/pervinco/Datasets/Vegetable Images"
train_set = CustomDataset(data_dir, "train", ToTensor())
print(len(train_set))

for idx in range(len(train_set)):
    image, label = train_set[idx]
    
    img = np.array(image.numpy()) * 255.0
    img = img.astype(np.uint8).transpose(1, 2, 0)
    
    cv2.imshow("image", img)
    cv2.waitKey(0)
    break

train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
for data in train_loader:
    print(data)
    break