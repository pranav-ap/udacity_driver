import math
from torch.utils.data import random_split, Dataset, DataLoader
import torchvision
import torchvision.transforms.functional as TF
import lightning.pytorch as pl

import os
import csv
from PIL import Image


class TripletDataset(Dataset):
    def __init__(self, images_dir:str, train:bool, samples, transform=None):
        super().__init__()
        self.images_dir = images_dir
        self.train = train
        self.samples = samples
        self.transform = transform
    
    def preprocess(self, image):
        image = Image.open(image)
        image = TF.crop(image, top=60, left=0, height=100, width=320)
        image = TF.resize(image, (128, 128))
        return image
    
    def __len__(self):
        length = len(self.samples)

        # H Flipped
        if self.train:
            length *= 2
                
            # Transformed
            if self.transform:
                length *= 2

        return length
    
    def __getitem__(self, index):
        """
        1N -> Original
        2N -> H Flipped
        3N -> Original Transformed
        4N -> H Flipped Transformed
        """
        section = math.ceil(index / len(self.samples))
        index = index % len(self.samples)
        image_center_path, image_left_path, image_right_path, steering_angle = self.samples[index]
        
        image_center = self.preprocess(image_center_path)
        image_left = self.preprocess(image_left_path)
        image_right = self.preprocess(image_right_path)
        
        if section == 2 or section == 4:
            image_center = TF.hflip(image_center)
            image_left = TF.hflip(image_left)
            image_right = TF.hflip(image_right)

            steering_angle = -steering_angle
        
        if (section == 3 or section == 4) and self.transform:
            image_center = self.transform(image_center)
            image_left = self.transform(image_left)
            image_right = self.transform(image_right)
        
        to_return = {
            'image_center': image_center,
            'image_left': image_left,
            'image_right': image_right,
            'steering_angle': steering_angle
        }

        return to_return


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
    
    def setup(self, stage: str):
        full_set = []

        # 1. Read CSV

        assert os.path.isfile(self.csv_path)
        
        with open(self.csv_path, 'r') as csvfile:
            data_reader = csv.reader(csvfile, delimiter=',')
            
            for row in data_reader:        
                sim_data = {
                    'image_center_path': row[0],
                    'image_left_path': row[1],
                    'image_right_path': row[2],
                    'steering_angle': row[3],
                    # 'throttle': row[4],
                    # 'brake': row[5],
                    # 'speed': row[6]
                }
                
                full_set.append(sim_data)
                # shuffle and remove some sim_data
        
        full_set = TripletDataset(self.images_dir, 
                                       train=False, # no augmentation for now
                                       samples=full_set)
        
        # 2. Split csv into three

        train_set_size = int(len(full_set) * 0.8)
        valid_set_size = int(len(full_set) * 0.2)
        test_set_size = len(full_set) - train_set_size - valid_set_size

        self.train_set, self.valid_set, self.test_set = random_split(full_set, 
                                                                     [train_set_size, 
                                                                      valid_set_size,
                                                                      test_set_size])
        
        # 3. Create three Datasets

        if stage == "fit" or stage is None:
            self.train_set = TripletDataset(self.images_dir, 
                                            train=True, 
                                            samples=self.train_set,
                                            transform=self.train_transform)
            
            self.valid_set = TripletDataset(self.images_dir, 
                                            train=True, 
                                            samples=self.valid_set,
                                            transform=self.test_transform)
            
        if stage == "test" or stage is None:
            self.test_set = TripletDataset(self.images_dir, 
                                           train=False, 
                                           samples=self.test_set,
                                           transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, 
                          batch_size=self.batch_size,
                          shuffle=True, # shuffle data every epoch?
                          num_workers=self.num_workers
                          )

    def val_dataloader(self):
        return DataLoader(self.valid_set, 
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers
                          )

    def test_dataloader(self):
        return DataLoader(self.test_set, 
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)
