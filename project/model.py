import torch
import torch.nn as nn
import torch.nn.functional as F
## Reference : https://github.com/vinceecws/SegNet_PyTorch

class SegNet(nn.Module):

    def __init__(self, BN_momentum=.5):
        super(SegNet, self).__init__()

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
        self.relu = nn.Tanh()

        self.MaxDe = nn.MaxUnpool2d(2, stride=2)

        self.ConvDe73 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe73 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe72 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe72 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe71 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe71 = nn.BatchNorm2d(512, momentum=BN_momentum)

        self.ConvDe63 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe63 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe62 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe62 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe61 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.BNDe61 = nn.BatchNorm2d(256, momentum=BN_momentum)

        self.ConvDe53 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe53 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe52 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe52 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe51 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe51 = nn.BatchNorm2d(256, momentum=BN_momentum)

        self.ConvDe43 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe43 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe42 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe42 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe41 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.BNDe41 = nn.BatchNorm2d(128, momentum=BN_momentum)

        self.ConvDe33 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNDe33 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvDe32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNDe32 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvDe31 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNDe31 = nn.BatchNorm2d(128, momentum=BN_momentum)

        self.ConvDe22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNDe22 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvDe21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.BNDe21 = nn.BatchNorm2d(64, momentum=BN_momentum)

        self.ConvDe12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNDe12 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvDe11 = nn.Conv2d(64, self.out_chn, kernel_size=3, padding=1)
        self.BNDe11 = nn.BatchNorm2d(self.out_chn, momentum=BN_momentum)

        self.threshold = torch.nn.Threshold(0,0)

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

        x = self.MaxDe(x, ind7, output_size=size6)
        x = F.relu(self.BNEn73(self.ConvDe73(x)))
        x = F.relu(self.BNDe72(self.ConvDe72(x)))
        x = F.relu(self.BNDe71(self.ConvDe71(x)))

        x = self.MaxDe(x, ind6, output_size=size5)
        x = F.relu(self.BNEn63(self.ConvDe63(x)))
        x = F.relu(self.BNDe62(self.ConvDe62(x)))
        x = F.relu(self.BNDe61(self.ConvDe61(x)))

        x = self.MaxDe(x, ind5, output_size=size4)
        x = F.relu(self.BNDe53(self.ConvDe53(x)))
        x = F.relu(self.BNDe52(self.ConvDe52(x)))
        x = F.relu(self.BNDe51(self.ConvDe51(x)))

        x = self.MaxDe(x, ind4, output_size=size3)
        x = F.relu(self.BNDe43(self.ConvDe43(x)))
        x = F.relu(self.BNDe42(self.ConvDe42(x)))
        x = F.relu(self.BNDe41(self.ConvDe41(x)))

        x = self.MaxDe(x, ind3, output_size=size2)
        x = F.relu(self.BNDe33(self.ConvDe33(x)))
        x = F.relu(self.BNDe32(self.ConvDe32(x)))
        x = F.relu(self.BNDe31(self.ConvDe31(x)))

        x = self.MaxDe(x, ind2, output_size=size1)
        x = F.relu(self.BNDe22(self.ConvDe22(x)))
        x = F.relu(self.BNDe21(self.ConvDe21(x)))

        x = self.MaxDe(x, ind1)
        x = F.relu(self.BNDe12(self.ConvDe12(x)))
        x = self.ConvDe11(x)
        return x
