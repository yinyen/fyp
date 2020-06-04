import numpy as np

from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision


__all__ = ['MNISTNet']
num_features = 512

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(VGGBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.relu(x)

        return output


class MNISTNetold(nn.Module):
    def __init__(self, args):
        super().__init__()
        num_features = 512
        self.pool = nn.MaxPool2d((2, 2))
        self.features = nn.Sequential(*[
            VGGBlock(1, 16, 16),
            self.pool,
            VGGBlock(16, 32, 32),
            self.pool,
            VGGBlock(32, 64, 64),
            self.pool,
        ])

        self.bn1 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout2d(0.5)
        self.fc = nn.Linear(3*3*64, num_features)
        self.bn2 = nn.BatchNorm1d(num_features)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        x = self.features(input)
        x = self.bn1(x)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        nn.Linear(3*3*64, num_features)
        x = self.fc(x)
        output = self.bn2(x)

        return output


class MNISTNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        num_features = 512
        self.pool = nn.MaxPool2d((2, 2))
        self.features = nn.Sequential(*[
            VGGBlock(3, 16, 16),
            # VGGBlock(3, 100, 100),
            self.pool,
            VGGBlock(16, 32, 32),
            self.pool,
            VGGBlock(32, 64, 64),
            self.pool,
        ])

        self.bn1 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout2d(0.5)
        # self.fc = nn.Linear(3*3*64*3, args.num_features)
        # self.fc = nn.Linear(50176, num_features) #224
        self.fc = nn.Linear(9216, num_features) #100
        self.bn2 = nn.BatchNorm1d(num_features)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        x = self.features(input)
        x = self.bn1(x)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        output = self.bn2(x)

        return output

