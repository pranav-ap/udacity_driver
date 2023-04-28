from torch.utils.data import random_split, Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import lightning.pytorch as pl

from PIL import Image
import csv
import random

STEERING_ANGLE_DELTA = 0.2
STEERING_PERTURB = 0.1


def preprocess(image):
    image = TF.crop(image, top=60, left=0, height=80, width=320)
    image = TF.resize(image, [128, 128])
    return image


class UdacitySimTripletDataset(Dataset):
    def __init__(self, samples, images_dir: str):
        super().__init__()
        self.samples = samples
        self.images_dir = images_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class UdacitySimDataset(Dataset):
    def __init__(self, samples, images_dir: str, train: bool, transform=None):
        super().__init__()
        self.samples = samples
        self.images_dir = images_dir
        self.train = train
        self.transform = transform

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

        steering_angle = self.samples[real_index]['steering_angle']

        # choose one of the three images randomly
        camera = random.choice(['frontal', 'left', 'right'])

        if camera == 'left':
            image = self.samples[real_index]['image_left']
            steering_angle = steering_angle + STEERING_ANGLE_DELTA
        elif camera == 'right':
            image = self.samples[real_index]['image_right']
            steering_angle = steering_angle - STEERING_ANGLE_DELTA
        else:
            image = self.samples[real_index]['image_center']

        # add noise to steering_angle
        steering_angle += random.uniform(-STEERING_PERTURB, STEERING_PERTURB)

        # Load and process image
        image = Image.open(self.images_dir + image)
        image = preprocess(image)

        # To counterbalance the left turn bias
        if steering_angle < 0 and random.random() <= 0.35:
            image = TF.hflip(image)
            steering_angle = -steering_angle

        if index >= len(self.samples) and self.transform:
            image = self.transform(image)

        # Note : PILToTensor does not scale values to 0 and 1
        image = T.PILToTensor()(image)

        return image, steering_angle


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

        self.test_set = None
        self.valid_set = None
        self.train_set = None

    def init_full_set(self):
        full_set = []

        with open(self.csv_path, 'r') as csvfile:
            data_reader = csv.reader(csvfile, delimiter=',')

            for row in data_reader:
                # Remove Most Straight Roads
                if random.random() > 0.15 and row[3] == '0':
                    continue

                # Example path - Desktop\track1data\IMG\center_2019_04_02_19_25_33_671.jpg

                full_set.append({
                    'image_center': row[0].split('\\')[-1],
                    'image_left': row[1].split('\\')[-1],
                    'image_right': row[2].split('\\')[-1],
                    'steering_angle': float(row[3]),
                })

        return full_set

    def setup(self, stage: str): # called once?
        full_set = self.init_full_set()
        full_set = UdacitySimTripletDataset(full_set, self.images_dir)

        self.train_set, self.valid_set, self.test_set = random_split(full_set,
                                                                    [0.7, 0.2, 0.1])

        # Create three Datasets

        if stage == "fit" or stage is None:
            self.train_set = UdacitySimDataset(
                images_dir=self.images_dir,
                train=True,
                samples=self.train_set,
                transform=self.train_transform
            )

            self.valid_set = UdacitySimDataset(
                images_dir=self.images_dir,
                train=True,
                samples=self.valid_set,
                transform=self.test_transform
            )

        if stage == "test" or stage is None:
            self.test_set = UdacitySimDataset(
                images_dir=self.images_dir,
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
