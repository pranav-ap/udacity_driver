import torch
import torch.nn as nn
import torch.nn.functional as F


class BabyHamiltonModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2),
            nn.ELU(),

            nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2),
            nn.BatchNorm2d(48),
            nn.ELU(),

            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ELU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ELU(),

            # kernel size is 9 - to reduce spatial dimension to 1x1 from 9x9
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=9),
            nn.ELU(),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1),
            nn.ELU(),

            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model_layers(x)
        x = torch.squeeze(x)
        return x

