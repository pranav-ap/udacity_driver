import torch
import torch.nn as nn
import torch.nn.functional as F


class BabyHamiltonModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=2)

        self.fc1 = nn.Linear(784, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # print('Size 1 : {0}'.format(x.size()))
        x = x.view(x.size(0), -1)
        # print('Size 2 : {0}'.format(x.size()))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        x = torch.squeeze(x)

        return x

# normalize to [-1, 1]
# x = x / 127.5 - 1

