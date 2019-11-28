from lib.modeling.unet.components import *


# code from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
class BaseUNet:
    def __init__(self, in_channels=3, n_classes=1):
        if n_classes == 2:
            n_classes = 1
        self.n_classes = n_classes
        self.inc = InConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up4 = Up(1024, 256)
        self.up3 = Up(512, 128)
        self.up2 = Up(256, 64)
        self.up1 = Up(128, 64)
        self.outc = OutConv(64, n_classes)
        self.up_features = [64, 128, 256, 512]
