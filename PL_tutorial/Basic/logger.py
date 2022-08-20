#use the weight and bias logger, here is 
#essentials
import os 
import warnings
warnings.filterwarnings("ignore")
#ds packagec
from sklearn.metrics import confusion_matrix
import seaborn as sns
import wandb
import matplotlib.pyplot as plt
import numpy as np
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
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.loggers import WandbLogger
#init the logger for experiment
wandb_logger = WandbLogger(project='lightning_tutorial')

#define the model
class Model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.linear(x)
        return x 
#customize callback for log image
#super() refers to super class 
#can us the call back to log 
#can also use call back to something else
class ImageCallBack(pl.Callback):
    def __init__(self, val_dataloader):
        super().__init__()
        self.val_dataloader = val_dataloader
        
    def on_validation_epoch_end(self, trainer, pl_module):
        #the module here is our model
        pred = []
        label = []
        for x, y in self.val_dataloader:
            yhat = torch.argmax(pl_module.model(x.reshape(BATCH_SIZE, -1)), dim=-1)
            pred.append(yhat.numpy())
            label.append(y.numpy())
        pred, label = np.array(pred).reshape(-1), np.array(label).reshape(-1)
        cf_matrix = confusion_matrix(pred, label)
        fig = plt.figure(figsize=(15, 10))
        plot = sns.heatmap(cf_matrix)
        trainer.logger.experiment.log({'Confusion matrix': wandb.Image(plot, caption="CM")})


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
        self.log("train_loss_iter", loss)
        return {'loss': loss}
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(BATCH_SIZE, -1)
        yhat = self.model(x)
        loss = nn.CrossEntropyLoss()(yhat, y)
        pred = torch.argmax(yhat, dim=-1)
        self.log("val_loss_iter", loss)
        return {'pred': pred, 'label': y}
    def training_epoch_end(self, outputs):
        all_preds = torch.stack([output['loss'] for output in outputs], dim=0)
        self.log("train_loss_epoch", torch.sum(all_preds, dim=0))
    def test_step(self, batch, batch_idx, on_epoch=True):
        x, y = batch
        x = x.reshape(BATCH_SIZE, -1)
        yhat = self.model(x)
        loss = nn.CrossEntropyLoss()(yhat, y)


        self.log("test_loss", loss)
        

    # def validation_epoch_end(self, outputs):
    #     pred = torch.cat([output['pred'] for output in outputs], dim=0)
    #     label = torch.cat([output['label'] for output in outputs], dim=0)
    #     cf_matrix = confusion_matrix(pred, label)
    #     plt.figure(figsize=(10, 5))
    #     plot = sns.heatmap(cf_matrix)
    #     self.logger.experiment.log({"Confusion Matrix": wandb.Image(plot, caption='test')})
    #format for optim config
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optim
#HYPER_PARAMS and SET UPs
input_dim = 784
num_classes = 10
BATCH_SIZE = 32
model = PL_module(input_dim=input_dim, num_classes=num_classes)
train_loader = DataLoader(train, batch_size=BATCH_SIZE, drop_last=True)
test_loader = DataLoader(test, batch_size=BATCH_SIZE, drop_last=True)
dev_loader = DataLoader(dev, batch_size=BATCH_SIZE, drop_last=True)
ckp_path = './checkpoint_saving/lightning_logs/version_0/checkpoints/epoch=3-step=6000.ckpt'
#basic early stopping
# callbacks = [EarlyStopping(monitor="val_loss_iter", mode='min', patience=3)]
# custom_callbacks = [Custom_EarlyStopping(monitor="val_loss_iter", mode='min', patience=3)]
# train 784 -> 10 
# define pl_module and trainer
# config the pl.trainer(), add the callbacks 
# trainer will fit to the model and the loader
# use fast_dev_run for trying to overfit one batch (training convention)
trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0, max_epochs=3, default_root_dir='./checkpoint_saving',  profiler='simple', logger=wandb_logger)
trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0, max_epochs=1, default_root_dir='./checkpoint_saving',  profiler='simple', logger=wandb_logger, callbacks=[ImageCallBack(dev_loader)])
trainer.fit(model, train_loader, dev_loader)
trainer.test(model, test_loader)
