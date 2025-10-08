import torch
import torch.nn as nn


# Simple UNet-like architecture suited for segmentation
class DoubleConv(nn.Module):
def __init__(self, in_ch, out_ch):
super().__init__()
self.net = nn.Sequential(
nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
nn.BatchNorm2d(out_ch),
nn.ReLU(inplace=True),
nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
nn.BatchNorm2d(out_ch),
nn.ReLU(inplace=True),
)
def forward(self, x):
return self.net(x)


class Down(nn.Module):
def __init__(self, in_ch, out_ch):
super().__init__()
self.pool_conv = nn.Sequential(
nn.MaxPool2d(2),
DoubleConv(in_ch, out_ch)
)
def forward(self, x):
return self.pool_conv(x)


class Up(nn.Module):
def __init__(self, in_ch, out_ch):
super().__init__()
self.up = nn.ConvTranspose2d(in_ch, in_ch//2, kernel_size=2, stride=2)
self.conv = DoubleConv(in_ch, out_ch)
def forward(self, x1, x2):
x1 = self.up(x1)
# pad if necessary
diffY = x2.size()[2] - x1.size()[2]
diffX = x2.size()[3] - x1.size()[3]
x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
diffY // 2, diffY - diffY // 2])
x = torch.cat([x2, x1], dim=1)
return self.conv(x)


class OutConv(nn.Module):
def __init__(self, in_ch, out_ch):
super().__init__()
self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
def forward(self, x):
return self.conv(x)


class JSDFloodNet(nn.Module):
def __init__(self, in_channels=5, n_classes=1, base_c=32):
super().__init__()
self.inc = DoubleConv(in_channels, base_c)
self.down1 = Down(base_c, base_c*2)
self.down2 = Down(base_c*2, base_c*4)
self.down3 = Down(base_c*4, base_c*8)
self.down4 = Down(base_c*8, base_c*8)
self.up1 = Up(base_c*16, base_c*4)
self.up2 = Up(base_c*8, base_c*2)
self.up3 = Up(base_c*4, base_c)
self.up4 = Up(base_c*2, base_c)
self.outc = OutConv(base_c, n_classes)


def forward(self, x):
x1 = self.inc(x)
x2 = self.down1(x1)
x3 = self.down2(x2)
x4 = self.down3(x3)
x5 = self.down4(x4)
x = self.up1(x5, x4)
x = self.up2(x, x3)
x = self.up3(x, x2)
x = self.up4(x, x1)
logits = self.outc(x)
return logits


if __name__ == '__main__':
model = JSDFloodNet(in_channels=5)
x = torch.randn(2,5,256,256)
y = model(x)
print(y.shape)
