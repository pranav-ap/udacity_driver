from model import BabyHamiltonModel

import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar

from utils import normalize_rgb_batch

"""
Training Utils
"""


def get_logger():
    from lightning.pytorch.loggers import CSVLogger
    logger = CSVLogger(save_dir='logs/', name='BabyHamilton')
    return logger

"""
Lightning Module
"""


class LightingBabyHamiltonModel(pl.LightningModule):
    def __init__(self,
                 model: BabyHamiltonModel,
                 learning_rate=0.1,
                 batch_size: int = 32):
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=['model'])

        self.example_input_array = torch.Tensor(batch_size, 3, 128, 128)

    def training_step(self, batch, batch_idx):
        x = normalize_rgb_batch(batch)
        # x = x.view(-1, 3, 128, 128)
        
        x_hat = self.model(x)

        train_loss = F.mse_loss(x_hat, x)
        self.log("train_loss", train_loss, prog_bar=True)

        return train_loss
    
    def evaluate(self, batch, stage=None):
        x = normalize_rgb_batch(batch)
        # x = x.view(-1, 3, 128, 128)
        
        x_hat = self.model(x)
        
        loss = F.mse_loss(x_hat, x)
        self.log(f"{stage}_loss", loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate)

        return optimizer

    def configure_callbacks(self):
        early_stop = EarlyStopping(
            monitor="val_acc",
            min_delta=0.00,
            patience=3,
            verbose=False,
            mode="max"
        )

        checkpoint = ModelCheckpoint(
            dirpath='checkpoints/',
            save_top_k=1,  # maximize the val acc
            mode='max',
            monitor='val_acc',
            save_last=True
        )
        
        progress_bar = TQDMProgressBar(
            refresh_rate=10
        )
    
        return [early_stop, checkpoint, progress_bar]
    

def start(model_path):
    pass
