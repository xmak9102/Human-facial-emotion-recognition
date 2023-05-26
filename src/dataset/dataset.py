#essentials
import os
#ds
import pandas as pd
import numpy as np
from PIL import Image
#dl
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torch.nn as nn 

DATA_DIR = "/media/data/chitb/study_zone/ML-_midterm_20212/data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

class EmotionSet(Dataset):
    def __init__(self, data_dir, mode=False, train=True):
        super().__init__()
        self.mode = mode
        self.data = pd.read_feather(data_dir)
        self.train = train
        if self.train:
            self.transform = transforms.Compose([
        transforms.RandomCrop(44),
        transforms.ToTensor()
        ])
        else:
            self.transform = transforms.Compose([
                                transforms.TenCrop(44), 
                                transforms.Lambda(lambda x: torch.stack([transforms.ToTensor()(i) for i in x])) #stack the tensors along new dimension
        ])

    def __getitem__(self, idx):
        #read
        img, label = self.data.iloc[idx]['pixel'], self.data.iloc[idx]['label']
        #transform
        img = np.array(img.tolist()).reshape(48, 48).astype(np.uint8)
        img = Image.fromarray(img)
        
        #transform
        img = self.transform(img)
        label = torch.tensor([label], dtype=torch.long)
        #if use pretrained model, have to switch to 3d
        if self.mode:
            if self.train:
                img = torch.concat([img, img, img], dim=0)
            else: 
                img = torch.concat([img, img, img], dim=1)
            
        return img, label
    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    # cal the mu and std
    data = EmotionSet(os.path.join(PROCESSED_DIR, "train.ftr"), train=True, mode=False)
    print("IMG", data[0][0].shape)
    print("label", data[0][1])
    
    # pretrained_model = torchvision.models.resnet18(pretrained=True)
    # print("model", pretrained_model)
    # pretrained_model.fc = nn.Linear(512, 7)
    # temp = np.array([0 for i in range(7)])
    # preds  = pretrained_model(torch.rand(320, 3, 44, 44))
    # print("PREDS", preds.shape)
    # # preds = preds.mean(dim=0)
    # # preds = torch.argmax(preds, dim=-1)
    # # print("PREDS vs LABELS", preds.item(), data[0][1].item())
        
    
    loader = DataLoader(data, batch_size=32)
    print(iter(loader).next()[0].view(32, -1).shape)
