from torch.utils.data import random_split, Dataset, DataLoader
import torchvision
import torchvision.transforms.functional as TF
import lightning.pytorch as pl

import csv
import random


class UdacitySimDataset(Dataset):
    def __init__(self, samples, images_dir: str, train: bool, transform=None):
        super().__init__()
        self.samples = samples
        self.images_dir = images_dir
        self.train = train
        self.transform = transform

    def preprocess(self, image):
        image = image / 127.5 - 1 # normalize to [-1, 1]
        image = TF.crop(image, top=60, left=0, height=80, width=320)
        image = TF.resize(image, (128, 128))
        return image

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
        index = index % len(self)
       
        steering_angle = self.samples[index]['steering_angle']
        image = self.samples[index]['image']
        image = torchvision.io.read_image(image)
        image = self.preprocess(image)

        if random.random() >= 0.5:
            image = TF.hflip(image)
            steering_angle = -steering_angle

        if index >= len(self) and self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'steering_angle': steering_angle
        }


class UdacitySimDataModule(pl.LightningDataModule):
    def __init__(self,
                 csv_path: str = "\data\driving_log.csv",
                 images_dir: str = "\data\IMG",
                 batch_size: int = 32,
                 num_workers: int = 1,
                 train_transform=None,
                 test_transform=None):
        super().__init__()
        self.csv_path = csv_path
        self.images_dir = images_dir

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = train_transform
        self.test_transform = test_transform

    def init_full_set(self):
        full_set = []
        steering_correction = 0.2

        with open(self.csv_path, 'r') as csvfile:
            data_reader = csv.reader(csvfile, delimiter=',')

            for row in data_reader:
                # Remove Most Straight Roads
                if random.random() > 0.15 and row[3] == '0':
                    continue

                full_set.append({
                    'image': row[0], # center
                    'steering_angle': float(row[3]),
                })

                full_set.append({
                    'image': row[1], # left
                    'steering_angle': float(row[3]) + steering_correction,
                })

                full_set.append({
                    'image': row[2], # right
                    'steering_angle': float(row[3]) - steering_correction,
                })

        return full_set

    def setup(self, stage: str):
        # get csv

        full_set = self.init_full_set()

        full_set = UdacitySimDataset(
            samples=full_set,
            images_dir=self.images_dir,
            train=False # no need augmentation here
        )

        # Split csv into three

        train_set_size = int(len(full_set) * 0.7)
        valid_set_size = int(len(full_set) * 0.2)
        test_set_size = int(len(full_set) * 0.1)

        self.train_set, self.valid_set, self.test_set = random_split(full_set,
                                                                     [train_set_size,
                                                                      valid_set_size,
                                                                      test_set_size])

        # Create three Datasets

        if stage == "fit" or stage is None:
            self.train_set = UdacitySimDataset(
                self.images_dir,
                train=True,
                samples=self.train_set,
                transform=self.train_transform
            )

            self.valid_set = UdacitySimDataset(
                self.images_dir,
                train=True,
                samples=self.valid_set,
                transform=self.test_transform
            )

        if stage == "test" or stage is None:
            self.test_set = UdacitySimDataset(
                self.images_dir,
                train=False,
                samples=self.test_set,
                transform=self.test_transform
            )

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.batch_size,
                          shuffle=True,  # shuffle data every epoch?
                          num_workers=self.num_workers
                          )

    def val_dataloader(self):
        return DataLoader(self.valid_set,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers
                          )

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

