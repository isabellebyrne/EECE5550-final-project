import torch
from torch import nn
import numpy as np

class OverFeat(nn.Module):
    def __init__(self):
        super(OverFeat, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(7,7), stride=2)
        self.maxp1 = nn.MaxPool2d(kernel_size=(3, 3), stride=3)

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(7,7), stride=1)
        self.maxp2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=1, padding=1)

        self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3,3), stride=1, padding=1)

        self.conv6 = nn.Conv2d(in_channels=1024, out_channels=4096, kernel_size=(3,3), stride=1, padding=1)
        self.maxp6 = nn.MaxPool2d(kernel_size=(3,3), stride=3)

        self.linear1 = nn.Linear(in_features=4096, out_features=4096)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxp1(x)
        x = self.conv2(x)
        x = self.maxp2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxp6(x)
        x = self.linear1(x)

        return x