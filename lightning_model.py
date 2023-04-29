import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from model import BabyHamiltonModel

"""
Lightning Utils
"""


def get_logger():
    from lightning.pytorch.loggers import CSVLogger
    logger = CSVLogger(save_dir='logs/', name='BabyHamilton')
    return logger

"""
Lightning Module
"""


class LightningBabyHamiltonModel(pl.LightningModule):
    def __init__(self,
                 model: BabyHamiltonModel,
                 learning_rate=0.1):
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=['model'])

    def training_step(self, batch, batch_idx):
        images, steering_angles = batch
        steering_angles_hat = self.model(images)

        train_loss = F.mse_loss(steering_angles_hat, steering_angles)
        self.log("train_loss", train_loss, prog_bar=True)

        return train_loss
    
    def evaluate(self, batch, stage=None):
        images, steering_angles = batch
        steering_angles_hat = self.model(images)

        loss = F.mse_loss(steering_angles_hat, steering_angles)
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
    
