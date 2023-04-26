from model import BabyHamiltonModel

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

from utils import normalize_rgb_batch

"""
Training Utils
"""

def get_logger():
    from lightning.pytorch.loggers import CSVLogger
    logger = CSVLogger(save_dir='logs/', name='BabyHamilton')
    return logger

def get_callbacks():   
    early_stop_callback = EarlyStopping(
        monitor="val_accuracy",
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode="max")

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        save_top_k=1,  # maximize the val acc
        mode='max',
        monitor='val_acc',
        save_last=True)

    callbacks= [
        checkpoint_callback, 
        early_stop_callback
    ]

    return callbacks


"""
Lightning Module
"""

class LightingBabyHamiltonModel(pl.LightningModule):
    def __init__(self, 
                 model:BabyHamiltonModel, 
                 learning_rate=0.1,
                 batch_size: int = 32):
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=['model'])
        
        self.example_input_array = torch.Tensor(batch_size, 1, 28, 28)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate)
        
        return optimizer

    def training_step(self, batch, batch_idx):
        batch = normalize_rgb_batch(batch)
        x = x.view(x.size(0), -1)
        
        z = self.encoder(x)
        x_hat = self.decoder(z)
        train_loss = F.mse_loss(x_hat, x)
        self.log("train_loss", train_loss, prog_bar=True)
        
        return train_loss

    def validation_step(self, batch, batch_idx):
        batch = normalize_rgb_batch(batch)
        x, y = batch
        x = x.view(x.size(0), -1)
        
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        self.log("val_loss", val_loss)
        
    def test_step(self, batch, batch_idx):
        batch = normalize_rgb_batch(batch)
        x, y = batch
        x = x.view(x.size(0), -1)
        
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)


def start(model_path):
    pass
