from lib.modeling.unet.components import *
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152


def res_unet_18(pre_train=True, n_classes=1):
    resnet = resnet18(pretrained=pre_train)
    return ResUNet(resnet=resnet, channel_configs=(64, 64, 128, 256, 512), n_classes=n_classes)


def res_unet_34(pre_train=True, n_classes=1):
    resnet = resnet34(pretrained=pre_train)
    return ResUNet(resnet=resnet, channel_configs=(64, 64, 128, 256, 512), n_classes=n_classes)


def res_unet_50(pre_train=True, n_classes=1):
    resnet = resnet50(pretrained=pre_train)
    return ResUNet(resnet=resnet, channel_configs=(64, 256, 512, 1024, 2048), n_classes=n_classes)


def res_unet_101(pre_train=True, n_classes=1):
    resnet = resnet101(pretrained=pre_train)
    return ResUNet(resnet=resnet, channel_configs=(64, 256, 512, 1024, 2048), n_classes=n_classes)


class ResUNet:
    def __init__(self, resnet, in_channels=3, n_classes=1, channel_configs=(64, 64, 128, 256, 512)):
        if n_classes == 2:
            n_classes = 1
        self.n_classes = n_classes
        self.inc = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        self.down1 = nn.Sequential(
            resnet.maxpool,
            resnet.layer1
        )
        self.inc = InConv(in_channels, channel_configs[0])  # channel_configs[0]
        self.down1 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
                                   resnet.layer1)  # channel_configs[1]
        self.down2 = resnet.layer2  # channel_configs[2]
        self.down3 = resnet.layer3  # channel_configs[3]
        self.down4 = resnet.layer4  # channel_configs[4]

        self.up4 = TernausUp(channel_configs[4], int(channel_configs[3] / 2), channel_configs[3], channel_configs[3])
        self.up3 = TernausUp(channel_configs[3], int(channel_configs[2] / 2), channel_configs[2], channel_configs[2])
        self.up2 = TernausUp(channel_configs[2], int(channel_configs[1] / 2), channel_configs[1], channel_configs[1])
        self.up1 = TernausUp(channel_configs[1], int(channel_configs[0] / 2), channel_configs[0], 32)
        self.outc = OutConv(32, n_classes)
        self.up_features = [channel_configs[1], channel_configs[2], channel_configs[3], channel_configs[4]]


if __name__ == '__main__':
    model = res_unet_18()
    print(model)
