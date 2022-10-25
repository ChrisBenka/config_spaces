import torch
import torch.nn as nn
import torch.nn.functional as F


class SegNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # layers
        # 1st encoder stage
        self.en_1_1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.en_1_2 = nn.Sequential(
            nn.Conv2d(64, 64, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # 2nd encoder stage
        self.en_2_1 = nn.Sequential(
            nn.Conv2d(64, 128, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.en_2_2 = nn.Sequential(
            nn.Conv2d(128, 128, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.en_2_1 = nn.Sequential(
            nn.Conv2d(64, 128, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.en_2_2 = nn.Sequential(
            nn.Conv2d(128, 128, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # 3rd encoder stage
        self.en_3_1 = nn.Sequential(
            nn.Conv2d(128, 256, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.en_3_2 = nn.Sequential(
            nn.Conv2d(256, 256, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.en_3_3 = nn.Sequential(
            nn.Conv2d(256, 256, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        # 4th encoder stage
        self.en_4_1 = nn.Sequential(
            nn.Conv2d(256, 512, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.en_4_2 = nn.Sequential(
            nn.Conv2d(512, 512, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.en_4_3 = nn.Sequential(
            nn.Conv2d(512, 512, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        # 5th encoder stage
        self.en_5_1 = nn.Sequential(
            nn.Conv2d(512, 512, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.en_5_2 = nn.Sequential(
            nn.Conv2d(512, 512, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.en_5_3 = nn.Sequential(
            nn.Conv2d(512, 512, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

    def forward(self, img):
        # 1st encoder stage
        x_1 = self.en_1_1(img)
        x_1 = self.en_1_2(x_1)
        x_1 = F.max_pool2d(x_1, kernel_size=2, stride=2)
        # 2nd encoder stage
        x_2 = self.en_2_1(x_1)
        x_2 = self.en_2_2(x_2)
        x_2 = F.max_pool2d(x_2, kernel_size=2, stride=2)
        # 3rd encoder stage
        x_3 = self.en_3_1(x_2)
        x_3 = self.en_3_2(x_3)
        x_3 = self.en_3_3(x_3)
        x_3 = F.max_pool2d(x_3, kernel_size=2, stride=2)
        # 4rd encoder stage
        x_4 = self.en_4_1(x_3)
        x_4 = self.en_4_2(x_4)
        x_4 = self.en_4_3(x_4)
        x_4 = F.max_pool2d(x_4, kernel_size=2, stride=2)
        # 5th encoder stage
        x_5 = self.en_4_1(x_4)
        x_5 = self.en_4_2(x_5)
        x_5 = self.en_4_3(x_5)
        x_5 = F.max_pool2d(x_5, kernel_size=2, stride=2)
