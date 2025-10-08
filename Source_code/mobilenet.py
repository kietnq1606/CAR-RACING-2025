import torch
import torch.nn as nn

class ResidualBottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, expansion_rate=6, stride=1):
        super().__init__()
        mid_ch = in_ch * expansion_rate

        # 1. Pointwise conv (expansion)
        self.pwconv = nn.Conv2d(in_ch, mid_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.act1 = nn.ReLU6(inplace=True)

        # 2. Depthwise conv
        self.dwconv = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=stride, padding=1, groups=mid_ch, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.act2 = nn.ReLU6(inplace=True)

        # 3. Pointwise conv (projection) → linear (no activation)
        self.pwconv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

        # Residual connection nếu stride=1 và in_ch == out_ch
        self.use_residual = (stride == 1 and in_ch == out_ch)

    def forward(self, x):
        out = self.act1(self.bn1(self.pwconv(x)))
        out = self.act2(self.bn2(self.dwconv(out)))
        out = self.bn3(self.pwconv2(out))

        if self.use_residual:
            out = x + out  # inverted residual connection
        return out

class UnetMobile(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, alpha=1):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, int(32*alpha), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(32*alpha)),
            nn.ReLU6(inplace=True)
        )

        self.enc2 = nn.Sequential(
            ResidualBottleneck(in_ch = int(32*alpha), out_ch = int(16*alpha), expansion_rate = 1, stride = 1),
            nn.BatchNorm2d(int(16*alpha))#192*320
        )
        self.enc3 = nn.Sequential(
            ResidualBottleneck(in_ch = int(16*alpha), out_ch = int(24*alpha), expansion_rate = 6, stride = 2),
            ResidualBottleneck(in_ch = int(24*alpha), out_ch = int(24*alpha), expansion_rate = 6, stride = 1),
            nn.BatchNorm2d(int(24*alpha))
        )
        self.enc4 = nn.Sequential(
            ResidualBottleneck(in_ch = int(24*alpha), out_ch = int(32*alpha), expansion_rate = 6, stride = 2),
            ResidualBottleneck(in_ch = int(32*alpha), out_ch = int(32*alpha), expansion_rate = 6, stride = 1),
            ResidualBottleneck(in_ch = int(32*alpha), out_ch = int(32*alpha), expansion_rate = 6, stride = 1),
            nn.BatchNorm2d(int(32*alpha))
        )
        self.enc5 = nn.Sequential(
            ResidualBottleneck(in_ch = int(32*alpha), out_ch = int(64*alpha), expansion_rate = 6, stride = 2),
            ResidualBottleneck(in_ch = int(64*alpha), out_ch = int(64*alpha), expansion_rate = 6, stride = 1),
            ResidualBottleneck(in_ch = int(64*alpha), out_ch = int(64*alpha), expansion_rate = 6, stride = 1),
            ResidualBottleneck(in_ch = int(64*alpha), out_ch = int(64*alpha), expansion_rate = 6, stride = 1),
            nn.BatchNorm2d(int(64*alpha))
        )
        self.enc6 = nn.Sequential(
            ResidualBottleneck(in_ch = int(64*alpha), out_ch = int(96*alpha), expansion_rate = 6, stride = 1),
            ResidualBottleneck(in_ch = int(96*alpha), out_ch = int(96*alpha), expansion_rate = 6, stride = 1),
            ResidualBottleneck(in_ch = int(96*alpha), out_ch = int(96*alpha), expansion_rate = 6, stride = 1),
            nn.BatchNorm2d(int(96*alpha))
        )
        self.enc7 = nn.Sequential(
            ResidualBottleneck(in_ch = int(96*alpha), out_ch = int(160*alpha), expansion_rate = 6, stride = 2),
            ResidualBottleneck(in_ch = int(160*alpha), out_ch = int(160*alpha), expansion_rate = 6, stride = 1),
            ResidualBottleneck(in_ch = int(160*alpha), out_ch = int(160*alpha), expansion_rate = 6, stride = 1),
            nn.BatchNorm2d(int(160*alpha))
        )
        self.enc8 = nn.Sequential(
            ResidualBottleneck(in_ch = int(160*alpha), out_ch = int(320*alpha), expansion_rate = 6, stride = 1),
            nn.BatchNorm2d(int(320*alpha))
        )
        # chia 2^5
        self.enc9 = nn.Sequential(
            nn.Conv2d(int(320*alpha), int(1280*alpha), kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(int(1280*alpha))
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(int(1280*alpha), int(320*alpha), 2, stride=2),
            nn.BatchNorm2d(int(320*alpha)),
            nn.ReLU6(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(int(320*alpha), int(160*alpha), 2, stride=2),
            nn.BatchNorm2d(int(160*alpha)),
            nn.ReLU6(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(int(160*alpha), int(64*alpha), 2, stride=2),
            nn.BatchNorm2d(int(64*alpha)),
            nn.ReLU6(inplace=True)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(int(64*alpha), int(32*alpha), 2, stride=2),
            nn.BatchNorm2d(int(32*alpha)),
            nn.ReLU6(inplace=True)
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(int(32*alpha),int(24*alpha), 2, stride=2),
            nn.BatchNorm2d(int(24*alpha)),
            nn.ReLU6(inplace=True)
        )
        self.final = nn.Conv2d(int(24*alpha), out_ch, kernel_size=1,padding=0, bias=False)
    def forward(self, x):
      out = self.enc1(x)
      out = self.enc2(out)
      out = self.enc3(out)
      out = self.enc4(out)
      out = self.enc5(out)
      out = self.enc6(out)
      out = self.enc7(out)
      out = self.enc8(out)
      out = self.enc9(out)
      out = self.up1(out)
      out = self.up2(out)
      out = self.up3(out)
      out = self.up4(out)
      out = self.up5(out)
      out = self.final(out)
      return out




