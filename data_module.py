import random

import lightning.pytorch as pl
import pandas as pd
import torchvision
from torch.utils.data import Dataset, DataLoader

STEERING_ANGLE_DELTA = 0.3
# STEERING_PERTURB = 0.1


class UdacitySimDataset(Dataset):
    def __init__(self, samples: pd.DataFrame, root_dir: str, train: bool, transform=None):
        super().__init__()
        self.samples = samples
        self.root_dir = root_dir
        self.train = train
        self.transform = transform

    def __len__(self):
        length = len(self.samples)

        # Double length if training and using augmentation
        # if self.train and self.transform:
        #     length *= 2

        return length

    def __getitem__(self, index):
        """
        1N -> Original
        2N -> Transformed
        """
        real_index = index # % len(self.samples)

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

        # add noise to steering_angle
        # steering_angle += random.uniform(-STEERING_PERTURB, STEERING_PERTURB)

        # Load and process image
        folder = self.samples.iloc[real_index]['folder']
        image = torchvision.io.read_image(self.root_dir + '\\preprocessed\\' + folder + image)

        # if index >= len(self.samples) and self.transform:
        #     image = self.transform(image)

        # if self.transform:
        #     image = self.transform(image)

        # Note : PILToTensor does not scale values to 0 and 1
        # image = T.PILToTensor()(image)

        return image, steering_angle


class UdacitySimDataModule(pl.LightningDataModule):
    def __init__(self,
                 csv_path_train: str = ".\\data\\driving_test_log.csv",
                 csv_path_test: str = ".\\data\\driving_train_log.csv",
                 root_dir: str = ".\\data\\",
                 batch_size: int = 32,
                 num_workers: int = 1,
                 train_transform=None,
                 test_transform=None):
        super().__init__()
        self.csv_path_train = csv_path_train
        self.csv_path_test = csv_path_test
        self.root_dir = root_dir

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
                root_dir=self.root_dir,
                train=True,
                samples=samples_train,
                transform=self.train_transform
            )

            self.valid_set = UdacitySimDataset(
                root_dir=self.root_dir,
                train=False,
                samples=samples_valid,
                transform=self.test_transform
            )

        if stage == "test":
            samples: pd.DataFrame = read_preprocessed_driving_csv(self.csv_path_test)

            self.test_set = UdacitySimDataset(
                root_dir=self.root_dir,
                train=False,
                samples=samples,
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
