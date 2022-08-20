#essentials
import os 
#torch
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
#lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

#define the model
class Model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.linear(x)
        return x 
#customize the early_stopping
class Custom_EarlyStopping(EarlyStopping):
    #turn off stop at val
    def on_validation_end(self, trainer, pl_module):
        pass
    def on_train_end(self, trainer, pl_module):
        self._run_early_stopping_check(trainer)


#datasets (with test and valid)
train = MNIST(root=os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
test = MNIST(root=os.getcwd(), download=True, transform=transforms.ToTensor(), train=False)
#split train, dev set
train_size = round(len(train)*0.8)
val_size = len(train) - train_size
train, dev = random_split(train, [train_size, val_size])
print("data and label", train[0][0].shape, train[0][1])


#lightning magic 
#PL modules
#in this part we use early stopping for supressing model from being overfitted
class PL_module(pl.LightningModule):
    def __init__(self, lr=1e-2, **kwargs):
        super().__init__()
        #our model defined here
        self.model = Model(**kwargs)
        #save to hparams
        self.save_hyperparameters()
    #format of pl in steps (train, dev, test)
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(BATCH_SIZE, -1)
        yhat = self.model(x)
        loss = nn.CrossEntropyLoss()(yhat, y)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(BATCH_SIZE, -1)
        yhat = self.model(x)
        loss = nn.CrossEntropyLoss()(yhat, y)
        self.log("val_loss", loss)
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(BATCH_SIZE, -1)
        yhat = self.model(x)
        loss = nn.CrossEntropyLoss()(yhat, y)
        self.log("test_loss", loss)
    #format for optim config
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optim
#HYPER_PARAMS and SET UPs
BATCH_SIZE = 32
model = PL_module(input_dim=784, num_classes=10)
train_loader = DataLoader(train, batch_size=BATCH_SIZE, drop_last=True)
test_loader = DataLoader(test, batch_size=BATCH_SIZE, drop_last=True)
dev_loader = DataLoader(dev, batch_size=BATCH_SIZE, drop_last=True)
ckp_path = './checkpoint_saving/lightning_logs/version_0/checkpoints/epoch=3-step=6000.ckpt'
#basic early stopping
callbacks = [EarlyStopping(monitor="val_loss", mode='min', patience=3)]
custom_callbacks = [Custom_EarlyStopping(monitor="val_loss", mode='min', patience=3)]
#train 784 -> 10 
#define pl_module and trainer
#config the pl.trainer(), add the callbacks 
#trainer will fit to the model and the loader

trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0, max_epochs=1, default_root_dir='./checkpoint_saving', callbacks=custom_callbacks)
trainer.fit(model, train_loader, dev_loader)
trainer.test(model, test_loader)
print("Callback", callbacks)