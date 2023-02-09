import torch
import torch.nn as nn
import torch.nn.functional as F


## Reference : https://github.com/vinceecws/SegNet_PyTorch


class ConvEncoder(nn.Module):
    def __init__(self, BN_momentum=.5):
        super(ConvEncoder, self).__init__()

        self.in_chn = 1
        self.out_chn = 1

        self.MaxEn = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.ConvEn11 = nn.Conv2d(self.in_chn, 64, kernel_size=3, padding=1)
        self.BNEn11 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvEn12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNEn12 = nn.BatchNorm2d(64, momentum=BN_momentum)

        self.ConvEn21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.BNEn21 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvEn22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNEn22 = nn.BatchNorm2d(128, momentum=BN_momentum)

        self.ConvEn31 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNEn31 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvEn32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNEn32 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvEn33 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNEn33 = nn.BatchNorm2d(128, momentum=BN_momentum)

        self.ConvEn41 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.BNEn41 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn42 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn42 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn43 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn43 = nn.BatchNorm2d(256, momentum=BN_momentum)

        self.ConvEn51 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn51 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn52 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn52 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn53 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn53 = nn.BatchNorm2d(256, momentum=BN_momentum)

        self.ConvEn61 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.BNEn61 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn62 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn62 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn63 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn63 = nn.BatchNorm2d(512, momentum=BN_momentum)

        self.ConvEn71 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn71 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn72 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn72 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn73 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn73 = nn.BatchNorm2d(512, momentum=BN_momentum)

    def forward(self, x):
        x = F.relu(self.BNEn11(self.ConvEn11(x)))
        x = F.relu(self.BNEn12(self.ConvEn12(x)))
        x, ind1 = self.MaxEn(x)
        size1 = x.size()

        x = F.relu(self.BNEn21(self.ConvEn21(x)))
        x = F.relu(self.BNEn22(self.ConvEn22(x)))
        x, ind2 = self.MaxEn(x)
        size2 = x.size()

        x = F.relu(self.BNEn31(self.ConvEn31(x)))
        x = F.relu(self.BNEn32(self.ConvEn32(x)))
        x = F.relu(self.BNEn33(self.ConvEn33(x)))
        x, ind3 = self.MaxEn(x)
        size3 = x.size()

        x = F.relu(self.BNEn41(self.ConvEn41(x)))
        x = F.relu(self.BNEn42(self.ConvEn42(x)))
        x = F.relu(self.BNEn43(self.ConvEn43(x)))
        x, ind4 = self.MaxEn(x)
        size4 = x.size()

        # Stage 5
        x = F.relu(self.BNEn51(self.ConvEn51(x)))
        x = F.relu(self.BNEn52(self.ConvEn52(x)))
        x = F.relu(self.BNEn53(self.ConvEn53(x)))
        x, ind5 = self.MaxEn(x)
        size5 = x.size()

        x = F.relu(self.BNEn61(self.ConvEn61(x)))
        x = F.relu(self.BNEn62(self.ConvEn62(x)))
        x = F.relu(self.BNEn63(self.ConvEn63(x)))
        x, ind6 = self.MaxEn(x)
        size6 = x.size()

        x = F.relu(self.BNEn71(self.ConvEn71(x)))
        x = F.relu(self.BNEn72(self.ConvEn72(x)))
        x = F.relu(self.BNEn73(self.ConvEn73(x)))
        x, ind7 = self.MaxEn(x)
        size7 = x.size()


class Model(nn.Module):
    def __init__(self, img_dim, num_dof):
        super(Model, self).__init__()
        self.conv_encoder = ConvEncoder()
        self.fc1 = nn.Linear(img_dim + num_dof, 1800)
        self.fc2 = nn.Linear(1800, 1024)
        self.fc3 = nn.Linear(1024, 800)
        self.fc4 = nn.Linear(800, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 128)
        self.fc7 = nn.Linear(128, 64)
        self.fc8 = nn.Linear(64, 1)
    def forward(self, input_img, degrees_of_freedom):
        x = torch.concat([torch.flatten(input_img), degrees_of_freedom])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return self.fc8(x),input_img
