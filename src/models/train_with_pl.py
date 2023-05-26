#essentials
import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.append("/media/data/chitb/study_zone/ML-_midterm_20212/src/dataset")
#ds
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
#dl
import torch.nn as nn
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from dataset import EmotionSet
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import precision_score, recall_score, f1_score
import torchmetrics
from sklearn.metrics import confusion_matrix
import wandb
import torchvision.transforms as transforms
import hydra 

pl.seed_everything(3)
DATA_DIR = "/media/data/chitb/study_zone/ML-_midterm_20212/data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
config_path = "/media/data/chitb/study_zone/ML-_midterm_20212/config"
max_epoch=30


class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=7, in_dim=1, out_dim=44):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, num_classes)
        self.relu2 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_dim, 3, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(3, in_dim, kernel_size=(3, 3))
        self.linear3 = nn.Linear(out_dim**2, num_classes)
        self.input_dim = input_dim
    
    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        return x
class EmotionModule(pl.LightningModule):
    def __init__(self, optim, epochs, tuning, batch_size=32, lr=1e-3, mode = "model", model=None):
        super().__init__()
        self.model = model
        self.optim = optim
        self.batch_size = batch_size
        self.save_hyperparameters()


    def training_step(self, batch, batch_idx):
        img, label = batch
    
        pred = self.model(img.float())
        pred = pred.reshape(pred.shape[0] , -1)
        label = label.reshape(-1)
        loss = nn.CrossEntropyLoss()(pred, label)
        pred = torch.argmax(pred, dim=-1)
        acc = (pred == label).sum()/pred.shape[0]
        self.log("train loss: ", loss)
        self.log("train accuracy: ", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        pred = self.model(img.float())
        pred = pred.reshape(pred.shape[0], -1)
        label = label.reshape(-1)
        dim = pred.shape[-1]
        loss = nn.CrossEntropyLoss()(pred, label)
        pred = torch.argmax(pred, dim=-1)
        acc = (pred == label).sum()/pred.shape[0]
        self.log("val loss: ", loss)
        self.log("val accuracy: ", acc)
        self.log('dim:', dim)
        return {'pred': pred, 'label': label}
    # #for 10 crop
    def test_step(self, batch, batch_idx):
        img, label = batch
        img, label = img.squeeze(0), label.squeeze(0)
        pred = self.model(img.float())
        pred = pred.reshape(1, 10, -1) #batch x crop x label
        pred = pred.mean(dim=1).reshape(1, -1)
        loss = nn.CrossEntropyLoss()(pred, label)
        pred = torch.argmax(pred, dim=-1)
        self.log("test loss: ", loss)
        self.log("test accuracy: ", int(pred.item() == label.item()))


        return {'pred': pred, 'label': label}
    def test_epoch_end(self, outputs):
        pred = torch.stack([i['pred'] for i in outputs], dim=0).cpu().numpy().reshape(-1)
        label = torch.stack([i['label'] for i in outputs], dim=0).cpu().numpy().reshape(-1)

        fig, _ = plt.subplots(figsize=(15, 6))
        cf_matrix = confusion_matrix(pred, label)
        
        sum = cf_matrix.sum(axis=1)
        cf_matrix = cf_matrix/sum
        plot = sns.heatmap(cf_matrix, annot=True)

        #test logger info
        self.logger.experiment.log({'Confusion matrix': wandb.Image(plot, caption="CM")})
        self.log("recall", recall_score(pred, label, average='weighted'))
        self.log("precision", precision_score(pred, label, average='weighted'))
        self.log("f1 score", f1_score(pred, label, average='weighted'))

    

        

    # def test_step(self, batch, batch_idx):
    #     img, label = batch
    #     # img = img.reshape(batch_size, -1).float()
    #     pred = self.model(img.float()).reshape(1, -1)

    #     label = label.reshape(-1)
    #     # loss = nn.CrossEntropyLoss()(pred, label)
    #     pred = torch.argmax(pred, dim=-1)
    #     # get the major voting 
    #     acc = (pred == label).sum()
    #     self.log("test accuracy: ", acc)
    #     return {'pred': pred, 'label': label}

    def validation_epoch_end(self, outputs): 
        pred = torch.cat([output['pred'] for output in outputs], dim=0)
        label = torch.cat([output['label'] for output in outputs], dim=0)


        #visualization
        pred, label = np.array(pred.detach().cpu().numpy()).reshape(-1), np.array(label.detach().cpu().numpy()).reshape(-1)
        cf_matrix = confusion_matrix(pred, label)
        fig = plt.figure(figsize=(15, 10))
        plot = sns.heatmap(cf_matrix, annot=True)
        self.logger.experiment.log({'Confusion matrix': wandb.Image(plot, caption="CM")})

    def configure_optimizers(self):
        if self.optim == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        lambda_lr = lambda epoch: 0.9**epoch
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-4)
        return [optimizer], scheduler
    def optimizer_step(
    self,
    epoch,
    batch_idx,
    optimizer,
    optimizer_idx,
    optimizer_closure,
    on_tpu=False,
    using_native_amp=False,
    using_lbfgs=False):
        # self.optimizer = optimizer.optimizer
        
        optimizer.step(closure=optimizer_closure)

        self.log("learning_rate", optimizer.param_groups[0]['lr'])
        # self.log("self learning_rate", self.optimizer.param_groups[0]['lr'])
@hydra.main(config_path=config_path, config_name='config')
def main(cfg):
    print(cfg.train.training.model)
    #get the model name and mode (using the pretrained or simple model)
    model, mode = cfg.train.training.model, cfg.train.training.mode                        
    pretrained_model = getattr(torchvision.models, model)(pretrained=True)
    if "vgg19" in model:
        pretrained_model.classifier[6] = nn.Linear(4096, 7) #for vgg
    elif "resnet18" in model:
        pretrained_model.fc = nn.Linear(512, 7) #for resnet

    #LIGHTNING MODULE
    #use pretrained or normal models:
    if cfg.train.training.pretrained:
        model = pretrained_model
    else:
        model = SimpleModel(input_dim=44**2, hidden_dim=512, num_classes=7)
    pl_module = EmotionModule(epochs=cfg.train.training.epochs, optim=cfg.train.optim.optimizer, lr=cfg.train.optim.lr, tuning=cfg.train.training.tuning, batch_size=cfg.train.loader.batch_size, mode=mode, model=model)
    
    #DATALOADERS
    #testing method: normal or 10-crop
    #data mode for 1d or 3d image: mode = False or True (pretrained models uses 3d image)
    val_mode = (cfg.val.method == "normal")
    test_mode = (cfg.test.method == "normal")
    train_data = EmotionSet(os.path.join(PROCESSED_DIR, "train.ftr"), mode=cfg.train.training.pretrained)
    val_data = EmotionSet(os.path.join(PROCESSED_DIR, "val.ftr"), mode=cfg.train.training.pretrained, train=val_mode)
    test_data = EmotionSet(os.path.join(PROCESSED_DIR, "test.ftr"), mode=cfg.train.training.pretrained, train=test_mode)
    #use both train and val for training at last time
    if not cfg.train.training.tuning:
        train_data = EmotionSet(os.path.join(PROCESSED_DIR, "data.ftr"), mode=cfg.train.training.pretrained)
    if cfg.train.training.raw:
        train_data = EmotionSet(os.path.join(PROCESSED_DIR, "raw.ftr"), mode=cfg.train.training.pretrained)
        
    train_dataloader = DataLoader(train_data, batch_size=cfg.train.loader.batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size=cfg.val.batch_size)
    test_dataloader = DataLoader(test_data, batch_size=cfg.test.batch_size)
    print("LOADER", iter(test_dataloader).next()[0].shape)

    #CALLBACKSss
    custom_callbacks = [LearningRateMonitor(logging_interval='step'), EarlyStopping(monitor='val loss: ', mode='min', patience=3), ModelCheckpoint(dirpath='/media/data/chitb/study_zone/ML-_midterm_20212/final_ckp', monitor='val loss: ', mode='min')]
    #epoch 9 -> best result
    #resume from checkpoint
    resume_path = cfg.train.training.resume
    load_from_ckp = cfg.train.training.load_from_ckp
    if load_from_ckp != False:
        pl_module = EmotionModule.load_from_checkpoint(load_from_ckp, lr=cfg.train.optim.lr)
    #TRAINER
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0, max_epochs=cfg.train.training.epochs, default_root_dir=ckp_dir, logger=wandb_logger, callbacks=custom_callbacks)
    
    #TRANING
    if cfg.train.training.train_mode:
        trainer.fit(pl_module, train_dataloader, val_dataloader, ckpt_path=cfg.train.training.resume)

    #TESTING
    trainer.test(pl_module, test_dataloader)

if __name__ == "__main__":
    #init the logger
    wandb_logger = WandbLogger(project='lightning_tutorial')
    ckp_dir = "/media/data/chitb/study_zone/ML-_midterm_20212/checkpoints"
    #magic 
    main()

    

