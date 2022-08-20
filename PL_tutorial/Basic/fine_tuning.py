#just use pretrained for input_model to module
#and the pretrained checkpoint 
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import os

import wandb 

import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
#dl
import torch.nn as nn
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import confusion_matrix
import wandb
import torchvision.transforms as transforms
import hydra 
from pytorch_lightning.callbacks import LearningRateMonitor
#define the model
class Model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.linear(x)
        return x 


#datasets
train = MNIST(root=os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
print("data and label", train[0][0].shape, train[0][1])


#lightning magic 
#PL modules
class PL_module(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        #our model defined here
        self.model = model
    #format of pl in steps (train, dev, test)
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(BATCH_SIZE, -1)
        yhat = self.model(x)
        loss = nn.CrossEntropyLoss()(yhat, y)
        self.log("train loss", loss)
        self.log("learning rate", self.trainer.optimizers[0]['lr'])
        return loss
    #format for optim config
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-4)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
#HYPER_PARAMS and SET UPs
BATCH_SIZE = 32
model = PL_module(Model(784, 10))
train_loader = DataLoader(train, batch_size=BATCH_SIZE)

#train 784 -> 10 
#define pl_module and trainer
#config the pl.trainer()
#trainer will fit to the model and the loaders
wandb_logger = WandbLogger(project='lightning_tutorial')
lr_monitor = LearningRateMonitor(logging_interval='step')
trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0, max_epochs=15, default_root_dir='/media/data/chitb/study_zone/ML-_midterm_20212/PL_tutorial/Basic/checkpoint_saving', logger=wandb_logger, callbacks=[lr_monitor])
trainer.fit(model=model, train_dataloaders=train_loader)