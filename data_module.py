import random

import lightning.pytorch as pl
import pandas as pd
import torchvision
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

STEERING_ANGLE_DELTA = 0.3
STEERING_JITTER = 0.08


class UdacitySimDataset(Dataset):
    def __init__(self, root_dir: str, samples: pd.DataFrame, train: bool, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.samples = samples
        self.train = train
        self.transform = transform

        self.image_dir = root_dir + 'train\\' if self.train else root_dir + 'test\\'

    def __len__(self):
        length = len(self.samples)

        # Double length if training and using augmentation
        if self.train and self.transform:
            length *= 2

        return length

    def __getitem__(self, index):
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
        steering_angle += random.uniform(-STEERING_JITTER, STEERING_JITTER)

        # Load and process image
        image = torchvision.io.read_image(self.image_dir + image)

        if random.random() <= 0.3:
            image = TF.hflip(image)
            steering_angle = -steering_angle

        if index >= len(self.samples) and self.train and self.transform:
            image = self.transform(image)

        return image, steering_angle


class UdacitySimDataModule(pl.LightningDataModule):
    def __init__(self,
                 root_dir: str,
                 batch_size: int = 32,
                 num_workers: int = 1,
                 train_transform=None,
                 test_transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.csv_path_train = root_dir + 'driving_train_log.csv'
        self.csv_path_test = root_dir + 'driving_test_log.csv'

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = train_transform
        self.test_transform = test_transform

        self.test_set = None
        self.valid_set = None
        self.train_set = None

    def setup(self, stage: str):
        # Create three Datasets
        from utils import read_preprocessed_driving_csv

        if stage == "fit":
            samples: pd.DataFrame = read_preprocessed_driving_csv(self.csv_path_train)

            from sklearn.model_selection import train_test_split
            samples_train, samples_valid = train_test_split(samples, train_size=0.8, shuffle=True)

            self.train_set = UdacitySimDataset(
                self.root_dir,
                samples=samples_train,
                train=True,
                transform=self.train_transform
            )

            self.valid_set = UdacitySimDataset(
                self.root_dir,
                samples=samples_valid,
                train=False,
                transform=self.test_transform
            )

        if stage == "test":
            samples: pd.DataFrame = read_preprocessed_driving_csv(self.csv_path_test)

            self.test_set = UdacitySimDataset(
                self.root_dir,
                samples=samples,
                train=False,
                transform=self.test_transform
            )

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.batch_size,
                          shuffle=True,  # shuffle data every epoch?
                          num_workers=self.num_workers,
                          persistent_workers=True,
                          pin_memory=True
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
