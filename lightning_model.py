import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger  # CSVLogger

from model import BabyHamiltonModel

"""
Lightning Utils
"""


def get_logger():
    logger = TensorBoardLogger(save_dir='lightning/logs/')
    # logger = CSVLogger(save_dir='lightning/logs/')
    return logger


"""
Lightning Module
"""


class LightningBabyHamiltonModel(pl.LightningModule):
    def __init__(self,
                 model: BabyHamiltonModel,
                 learning_rate=0.01):
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
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
        )

        return optimizer

    def configure_callbacks(self):
        early_stop = EarlyStopping(
            monitor="val_loss",
            mode="min",
            min_delta=0.00,
            patience=3,
            verbose=False,
        )

        checkpoint = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            dirpath='lightning/checkpoints/',
            save_top_k=1,
            save_last=True
        )

        progress_bar = TQDMProgressBar()

        return [early_stop, checkpoint, progress_bar]

