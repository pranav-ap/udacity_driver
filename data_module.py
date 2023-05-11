import random
import os
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torchvision
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from utils import read_preprocessed_driving_csv

STEERING_ANGLE_DELTA = 0.3
STEERING_JITTER = 0.08


class UdacitySimDataset(Dataset):
    def __init__(self, root_dir: str, samples: pd.DataFrame, stage: str, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.samples = samples
        self.stage = stage
        self.transform = transform

        self.image_dir = os.path.join(
            root_dir,
            'train' if (stage == 'train' or stage == 'valid') else 'test'
        )

    def __len__(self):
        length = len(self.samples)

        # Double length if training as we use augmentation
        if self.stage == 'train' and self.transform:
            length *= 2

        return length

    def get_item_train(self, index):
        """
        1N -> Original
        2N -> Transformed
        """
        real_index = index % len(self.samples)
        steering_angle = self.samples.iloc[real_index]['steering_angle']

        # choose one of the three images randomly
        camera = random.choice(['center', 'left', 'right'])

        if camera == 'left':
            image = self.samples.iloc[real_index]['left']
            steering_angle = steering_angle + STEERING_ANGLE_DELTA
        elif camera == 'right':
            image = self.samples.iloc[real_index]['right']
            steering_angle = steering_angle - STEERING_ANGLE_DELTA
        else:
            image = self.samples.iloc[real_index]['center']

        # JITTER : add noise to steering_angle
        # steering_angle += random.uniform(-STEERING_JITTER, STEERING_JITTER)

        # Load and process image
        path = os.path.join(self.image_dir, image)
        image = torchvision.io.read_image(path)

        if random.random() < 0.3:
            image = TF.hflip(image)
            steering_angle = -steering_angle

        if self.transform and index >= len(self.samples):
            image = self.transform(image)
        else:
            image = image / 127.5 - 1  # normalize to [-1, 1]

        return image, steering_angle

    def get_item_valid_test(self, index):
        steering_angle = self.samples.iloc[index]['steering_angle']

        image = self.samples.iloc[index]['center']
        path = os.path.join(self.image_dir, image)
        image = torchvision.io.read_image(path)

        if self.transform:
            image = self.transform(image)
        else:
            image = image / 127.5 - 1  # normalize to [-1, 1]

        return image, steering_angle

    def __getitem__(self, index):
        image, steering_angle = self.get_item_train(index) if self.stage == 'train' else self.get_item_valid_test(index)

        image = image.float()  # tensor float 32
        steering_angle = steering_angle.astype(np.float32)  # np scalar float 32

        return image, steering_angle


class UdacitySimDataModule(pl.LightningDataModule):
    def __init__(self,
                 root_dir: str,
                 train_csv: str,
                 test_csv: str,
                 batch_size: int = 32,
                 num_workers: int = 1,
                 train_transform=None,
                 test_transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.csv_path_train = os.path.join(root_dir, train_csv)
        self.csv_path_test = os.path.join(root_dir, test_csv)

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = train_transform
        self.test_transform = test_transform

        self.train_set = None
        self.valid_set = None
        self.test_set = None

    def setup(self, stage: str):
        if stage == "fit":
            samples: pd.DataFrame = read_preprocessed_driving_csv(self.csv_path_train)
            samples_train, samples_valid = train_test_split(samples, train_size=0.8, shuffle=True)

            self.train_set = UdacitySimDataset(
                self.root_dir,
                samples=samples_train,
                stage='train',
                transform=self.train_transform
            )

            self.valid_set = UdacitySimDataset(
                self.root_dir,
                samples=samples_valid,
                stage='valid',
                transform=self.test_transform
            )

        if stage == "test":
            samples: pd.DataFrame = read_preprocessed_driving_csv(self.csv_path_test)

            self.test_set = UdacitySimDataset(
                self.root_dir,
                samples=samples,
                stage='test',
                transform=self.test_transform
            )

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.batch_size,
                          shuffle=True,  # shuffle data every epoch?
                          num_workers=self.num_workers,
                          persistent_workers=True,
                          pin_memory=True,
                          )

    def val_dataloader(self):
        return DataLoader(self.valid_set,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          persistent_workers=True,
                          pin_memory=True
                          )

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          persistent_workers=True,
                          pin_memory=True
                          )
