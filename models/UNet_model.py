import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

# Unet model derived from https://github.com/milesial/Pytorch-UNet
class double_conv(nn.Module):
    '''(conv => ELU) * 2'''
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is NCHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class interpDownConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.downConv = double_conv(in_ch,out_ch)

    def forward(self,x):
        x = F.interpolate(x, scale_factor=0.5)
        return self.downConv(x)

class UNetGenerator(nn.Module):
    def __init__(self, n_channels=6):
        super().__init__()
        self.inc = double_conv(n_channels, 16)
#         self.down1 = interpDownConv(16, 32)
#         self.down2 = interpDownConv(32, 32)
#         self.up1 = up(64, 16)
#         self.up2 = up(32, 16)
        self.down1 = interpDownConv(16, 32)
        self.down2 = interpDownConv(32, 64)
        self.down3 = interpDownConv(64, 64)
        self.up1 = up(128, 32)
        self.up2 = up(64, 16)
        self.up3 = up(32, 16)
        self.outc = double_conv(16,3)

    def forward(self, x):
        x = torch.cat(x, dim=1)
        x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x = self.up1(x3, x2)
#         x = self.up2(x, x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return x

class UNetDiscriminator(nn.Module):
    def __init__(self, n_channels=3, height=512, width=288, hidden_size=300):
        super().__init__()
        self.inc = double_conv(n_channels, 16)
#         self.down1 = interpDownConv(16, 32)
#         self.down2 = interpDownConv(32, 32)
#         self.up1 = up(64, 16)
#         self.up2 = up(32, 16)
        self.down1 = interpDownConv(16, 32)
        self.down2 = interpDownConv(32, 64)
        self.down3 = interpDownConv(64, 64)
        self.up1 = up(128, 32)
        self.up2 = up(64, 16)
        self.up3 = up(32, 16)
        self.final = nn.Sequential(
            double_conv(16, 3),
            Flatten(),
            nn.Linear(height * width * 3, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1))

    def forward(self, x):
        x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x = self.up1(x3, x2)
#         x = self.up2(x, x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.final(x)
