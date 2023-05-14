import torch
import torch.nn as nn
import torch.nn.functional as F


class BabyHamiltonModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3)

        self.fc1 = nn.Linear(in_features=7744, out_features=100)  # 7744, 5184
        self.fc2 = nn.Linear(in_features=100, out_features=30)
        self.fc3 = nn.Linear(in_features=30, out_features=1)

    def forward(self, x):
        # print('Size 0 : {0}'.format(x.size()))

        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        # print('Size 1 : {0}'.format(x.size()))
        x = x.view(x.size(0), -1)
        # print('Size 2 : {0}'.format(x.size()))

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.tanh(self.fc3(x))

        # print('Size 3 : {0}'.format(x.size()))
        x = torch.squeeze(x)
        # print('Size 4 : {0}'.format(x.size()))

        return x

