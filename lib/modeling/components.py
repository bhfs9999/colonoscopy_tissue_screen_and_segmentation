from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_ch, inter_ch=64, se_module=False):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(inter_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_ch, in_ch, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
        )
        self.relu = nn.ReLU(inplace=True)

        self.se_module = se_module
        if se_module:
            self.se = SELayer(in_ch)

    def forward(self, x):
        identity = x

        x = self.conv(x)
        if self.se_module:
            x = self.se(x)
        x += identity
        y = self.relu(x)
        return y


class SELayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SELayer, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # b, c, h, w
        y = self.squeeze(x).view(b, c)  # b, c
        weight = self.excitation(y).view(b, c, 1, 1)  # b, c, 1, 1
        return x * weight.expand_as(x)
