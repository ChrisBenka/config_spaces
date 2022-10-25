import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels):
        self.in_channels = in_channels
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
        x_1, x_1_idxs = F.max_pool2d(x_1, kernel_size=2, stride=2, return_indices=True)
        # 2nd encoder stage
        x_2 = self.en_2_1(x_1)
        x_2 = self.en_2_2(x_2)
        x_2, x_2_idxs = F.max_pool2d(x_2, kernel_size=2, stride=2, return_indices=True)
        # 3rd encoder stage
        x_3 = self.en_3_1(x_2)
        x_3 = self.en_3_2(x_3)
        x_3 = self.en_3_3(x_3)
        x_3, x_3_idxs = F.max_pool2d(x_3, kernel_size=2, stride=2, return_indices=True)
        # 4rd encoder stage
        x_4 = self.en_4_1(x_3)
        x_4 = self.en_4_2(x_4)
        x_4 = self.en_4_3(x_4)
        x_4, x_4_idxs = F.max_pool2d(x_4, kernel_size=2, stride=2, return_indices=True)
        # 5th encoder stage
        x_5 = self.en_4_1(x_4)
        x_5 = self.en_4_2(x_5)
        x_5 = self.en_4_3(x_5)
        x_5, x_5_idxs = F.max_pool2d(x_5, kernel_size=2, stride=2, return_indices=True)
        return x_5, (x_1_idxs, x_2_idxs, x_3_idxs, x_4_idxs, x_5_idxs)


class Decoder(nn.Module):
    def __init__(self, out_channels):
        # DECODER
        # decoder layer 5 (corresponds to encoder layer 5)
        self.out_channels = out_channels
        self.de_5_1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.de_5_2 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.de_5_3 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        # Decoder layer 4 (corresponds to encoder layer 4)
        self.de_4_1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.de_4_2 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.de_4_3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Decoder layer 3 (corresponds to encoder layer 3)
        self.de_3_1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.de_3_2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.de_3_3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # Decoder layer 2 (corresponds to encoder layer 2)
        self.de_2_1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.de_2_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Decoder layer 1 (corresponds to encoder layer 1)
        self.de_1_1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.de_1_2 = nn.Sequential(
            nn.ConvTranspose2d(64, self.out_channels, stride=1, kernel_size=3, padding=1),
        )

    def forward(self, encoded, indices):
        x_5_decoded = F.max_unpool2d(encoded, kernel_size=3, stride=2, indices=indices[-1], output_size=512)
        x_5_decoded = self.de_5_1(x_5_decoded)
        x_5_decoded = self.de_5_2(x_5_decoded)
        x_5_decoded = self.de_5_3(x_5_decoded)

        x_4_decoded = F.max_unpool2d(x_5_decoded, kernel_size=3, stride=2, indices=indices[-2], output_size=512)
        x_4_decoded = self.de_4_1(x_4_decoded)
        x_4_decoded = self.de_4_2(x_4_decoded)
        x_4_decoded = self.de_4_3(x_4_decoded)

        x_3_decoded = F.max_unpool2d(x_4_decoded, kernel_size=3, stride=2, indices=indices[-3], output_size=256)
        x_3_decoded = self.de_3_1(x_3_decoded)
        x_3_decoded = self.de_3_2(x_3_decoded)
        x_3_decoded = self.de_3_3(x_3_decoded)

        x_2_decoded = F.max_unpool2d(x_3_decoded, kernel_size=3, stride=2, indices=indices[-4], output_size=128)
        x_2_decoded = self.de_2_1(x_2_decoded)
        x_2_decoded = self.de_2_2(x_2_decoded)

        x_1_decoded = F.max_unpool2d(x_2_decoded, kernel_size=3, stride=2, indices=indices[0], output_size=64)
        x_1_decoded = self.de_1_1(x_1_decoded)
        x_1_decoded = self.de_1_2(x_1_decoded)

        return x_1_decoded


class SegNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(out_channels)

    def forward(self, image):
        encoded, indxs_tuple = self.encoder(image)
        decoded = self.decoder(encoded, indxs_tuple)
        return decoded
