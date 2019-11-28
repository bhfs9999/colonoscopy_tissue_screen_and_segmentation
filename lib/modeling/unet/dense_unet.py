from lib.modeling.unet.components import *
from torchvision.models import densenet121


def dense_unet_121(pre_train=True, n_classes=1):
    densenet = densenet121(pretrained=pre_train)
    return DenseUNet(densenet=densenet, n_classes=n_classes)


class DenseUNet:
    def __init__(self, densenet, in_channels=3, n_classes=1):
        if n_classes == 2:
            n_classes = 1
        self.n_classes = n_classes
        self.inc = InConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
                                   densenet.features.denseblock1,
                                   SingleConv(256, 128))

        self.down2 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
                                   densenet.features.denseblock2,
                                   SingleConv(512, 256))

        self.down3 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
                                   densenet.features.denseblock3,
                                   SingleConv(1024, 512))

        self.down4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
                                   densenet.features.denseblock4,
                                   SingleConv(1024, 512))

        self.up4 = Up(1024, 256)
        self.up3 = Up(512, 128)
        self.up2 = Up(256, 64)
        self.up1 = Up(128, 64)
        self.outc = OutConv(64, n_classes)
        self.up_features = [64, 128, 256, 512]