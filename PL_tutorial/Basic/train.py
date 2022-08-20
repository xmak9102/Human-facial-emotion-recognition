import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import os 

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
        return loss
    #format for optim config
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optim
#HYPER_PARAMS and SET UPs
BATCH_SIZE = 32
model = PL_module(Model(784, 10))
train_loader = DataLoader(train, batch_size=BATCH_SIZE)

#train 784 -> 10 
#define pl_module and trainer
#config the pl.trainer()
#trainer will fit to the model and the loader

trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0)
trainer.fit(model=model, train_dataloaders=train_loader)
print("????", trainer.optimizers)