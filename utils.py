import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# Other


def preprocess(image):
    image = TF.crop(image, top=60, left=0, height=80, width=320)
    image = TF.resize(image, [128, 128], antialias=True)
    return image


def all_paths_exist(paths):
    return all([os.path.exists(path) for path in paths])


def read_preprocessed_driving_csv(csv_path: str):
    with open(csv_path, 'r') as f:
        df = pd.read_csv(csv_path)
        return df


def extract_file_name(path):
    return path.split('\\')[-1]


# PIL Normalization - better term here is Standardization


def normalize_rgb_get_mean_std(x):
    mean_per_channel = x.mean(dim=[1, 2])
    std_per_channel = x.std(dim=[1, 2])

    y = TF.normalize(x,
                     mean=mean_per_channel,
                     std=std_per_channel)

    return y, mean_per_channel, std_per_channel


def normalize_rgb(x):
    mean_per_channel = x.mean(dim=[1, 2])
    std_per_channel = x.std(dim=[1, 2])

    y = TF.normalize(x,
                     mean=mean_per_channel,
                     std=std_per_channel)

    return y


def normalize_rgb_batch(batch):
    mean_per_channel = batch.view(3, -1).mean(dim=1)
    std_per_channel = batch.view(3, -1).std(dim=1)

    y = TF.normalize(batch,
                     mean=mean_per_channel,
                     std=std_per_channel)

    return y


def denormalize_rgb(x, mean_per_channel, std_per_channel):
    mean_per_channel = mean_per_channel.view(-1, 1, 1)
    std_per_channel = std_per_channel.view(-1, 1, 1)
    y = x * std_per_channel + mean_per_channel
    return y


def denormalize_rgb_batch(x, mean_per_channel, std_per_channel):
    """
    x : normalized tensor image batch
    """
    # 3, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean_per_channel, std_per_channel):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)


# Visualization


def show_tensor_images(images):
    fix, axs = plt.subplots(ncols=len(images), squeeze=False)
    for i, img in enumerate(images):
        img = T.ToPILImage()(img.to('cpu'))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

