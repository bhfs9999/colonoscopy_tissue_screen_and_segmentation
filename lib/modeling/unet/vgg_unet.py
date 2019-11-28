from lib.modeling.unet.components import *
from lib.modeling.components import SELayer
from torchvision.models import vgg11, vgg16, vgg16_bn, vgg19


def unet11(pre_train=True, se=False, residual=False, **kwargs):
    """
    pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with VGG11
            carvana - all weights are pre-trained on
                Kaggle: Carvana dataset https://www.kaggle.com/c/carvana-image-masking-challenge
    """
    vgg_model = vgg11(pretrained=pre_train)

    if se:
        return UNet11SE(backbone=vgg_model, **kwargs)
    if residual:
        return ResUNet11SE(backbone=vgg_model, **kwargs)

    return UNet11(backbone=vgg_model, **kwargs)


def unet16(pre_train=True, bn=False, layer_4=False, **kwargs):
    if bn:
        vgg_model = vgg16_bn(pretrained=pre_train)
        return UNet16BN(backbone=vgg_model, **kwargs)

    else:
        vgg_model = vgg16(pretrained=pre_train)

        if layer_4:
            return UNet16Layer4(backbone=vgg_model, **kwargs)

        else:
            return UNet16(backbone=vgg_model, **kwargs)


def unet19(pre_train=True, layer_4=False, **kwargs):
    vgg_model = vgg19(pretrained=pre_train)
    if layer_4:
        return UNet19Layer4(backbone=vgg_model, **kwargs)
    else:
        return UNet19(backbone=vgg_model, **kwargs)


class UNet11:
    def __init__(self, backbone, n_classes=1, num_filters=32):
        """
        :param backbone:
        :param n_classes:
        :param num_filters:
        """
        super().__init__()
        if n_classes == 2:
            n_classes = 1

        self.n_classes = n_classes
        self.encoder = backbone.features

        self.inc = self.encoder[0:2]
        self.down1 = self.encoder[2:5]
        self.down2 = self.encoder[5:10]
        self.down3 = self.encoder[10:15]
        self.down4 = self.encoder[15:20]
        self.down5 = nn.Sequential(
            self.encoder[20],  # MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            ConvRelu(512, num_filters * 16)
        )

        self.up5 = TernausUp(num_filters * 16, num_filters * 8, num_filters * 16, num_filters * 16)
        self.up4 = TernausUp(num_filters * 16, num_filters * 8, num_filters * 16, num_filters * 16)
        self.up3 = TernausUp(num_filters * 16, num_filters * 4, num_filters * 8, num_filters * 8)
        self.up2 = TernausUp(num_filters * 8, num_filters * 2, num_filters * 4, num_filters * 4)
        self.up1 = TernausUp(num_filters * 4, num_filters * 1, num_filters * 2, num_filters)
        self.outc = nn.Conv2d(num_filters, n_classes, kernel_size=1)
        self.up_features = [128, 256, 512, 512, 512]


class UNet11SE:
    """
    this class only add se layer after each encoder of unet11
    it should be merged to unet11 class
    """
    def __init__(self, backbone, n_classes=1, num_filters=32):
        """
        :param backbone:
        :param n_classes:
        :param num_filters:
        """
        super().__init__()
        if n_classes == 2:
            n_classes = 1
        self.n_classes = n_classes
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = backbone.features

        self.relu = self.encoder[1]
        self.inc = nn.Sequential(
            self.encoder[0],   # Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu,
            SELayer(channels=64))
        self.down1 = nn.Sequential(
            self.pool,
            self.encoder[3],  # Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu,
            SELayer(channels=128))
        self.down2 = nn.Sequential(
            self.pool,
            self.encoder[6],  # Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu,
            SELayer(channels=256),
            self.encoder[8],  # Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu,
            SELayer(channels=256))
        self.down3 = nn.Sequential(
            self.pool,
            self.encoder[11],  # Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu,
            SELayer(channels=512),
            self.encoder[13],  # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu,
            SELayer(channels=512))
        self.down4 = nn.Sequential(
            self.pool,
            self.encoder[16],  # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu,
            SELayer(channels=512),
            self.encoder[18],  # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu,
            SELayer(channels=512))
        self.down5 = nn.Sequential(
            self.pool,
            # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ConvRelu(num_filters * 16, num_filters * 16),
            SELayer(channels=512))

        self.up5 = TernausUp(num_filters * 16, num_filters * 8, num_filters * 16, num_filters * 16)
        self.up4 = TernausUp(num_filters * 16, num_filters * 8, num_filters * 16, num_filters * 16)
        self.up3 = TernausUp(num_filters * 16, num_filters * 4, num_filters * 8, num_filters * 8)
        self.up2 = TernausUp(num_filters * 8, num_filters * 2, num_filters * 4, num_filters * 4)
        self.up1 = TernausUp(num_filters * 4, num_filters * 1, num_filters * 2, num_filters)
        self.outc = nn.Conv2d(num_filters, n_classes, kernel_size=1)
        self.up_features = [128, 256, 512, 512, 512]


class ResUNet11SE:
    """
    add identical path in each encoder, add se block
    """
    def __init__(self, backbone, n_classes=1, num_filters=32):
        """
        :param backbone:
        :param n_classes:
        :param num_filters:
        """
        super().__init__()
        if n_classes == 2:
            n_classes = 1
        self.n_classes = n_classes
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = backbone.features

        self.relu = self.encoder[1]
        # Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inc = nn.Sequential(
            self.encoder[0],  # Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu,
            SELayer(channels=64))
        self.down1 = nn.Sequential(
            self.pool,
            self.encoder[3],  # Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu,
            SELayer(channels=128))
        self.down2 = nn.Sequential(
            self.pool,
            self.encoder[6],  # Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu,
            SELayer(channels=256),
            # Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ResidualEncoder(self.encoder[8], channels=256, se_block=True),
            )
        self.down3 = nn.Sequential(
            self.pool,
            self.encoder[11],  # Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu,
            SELayer(channels=512),
            # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ResidualEncoder(self.encoder[13], channels=512, se_block=True),
            )
        self.down4 = nn.Sequential(
            self.pool,
            # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ResidualEncoder(self.encoder[16], channels=512, se_block=True),
            # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ResidualEncoder(self.encoder[18], channels=512, se_block=True),
            )
        self.down5 = nn.Sequential(
            self.pool,
            # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ResidualEncoder(nn.Conv2d(num_filters * 16, num_filters * 16, kernel_size=3, padding=1),
                            channels=512, se_block=True),
            )

        self.up5 = TernausUp(num_filters * 16, num_filters * 8, num_filters * 16, num_filters * 16)
        self.up4 = TernausUp(num_filters * 16, num_filters * 8, num_filters * 16, num_filters * 16)
        self.up3 = TernausUp(num_filters * 16, num_filters * 4, num_filters * 8, num_filters * 8)
        self.up2 = TernausUp(num_filters * 8, num_filters * 2, num_filters * 4, num_filters * 4)
        self.up1 = TernausUp(num_filters * 4, num_filters * 1, num_filters * 2, num_filters)
        self.outc = nn.Conv2d(num_filters, n_classes, kernel_size=1)
        self.up_features = [128, 256, 512, 512, 512]


class UNet16:
    def __init__(self, backbone, n_classes=1, num_filters=32):
        """
        :param backbone: vgg16
        :param n_classes:
        :param num_filters:
            False - no pre-trained network used
            True - encoder pre-trained with VGG16
        """
        super().__init__()
        if n_classes == 2:
            n_classes = 1

        self.n_classes = n_classes
        self.encoder = backbone.features

        self.inc = self.encoder[0:4]
        self.down1 = self.encoder[4:9]
        self.down2 = self.encoder[9:16]
        self.down3 = self.encoder[16:23]
        self.down4 = self.encoder[23:30]
        self.down5 = nn.Sequential(
            self.encoder[30],  # MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            ConvRelu(512, num_filters * 16)
        )

        self.up5 = TernausUp(num_filters * 16, num_filters * 8, num_filters * 16, num_filters * 16)
        self.up4 = TernausUp(num_filters * 16, num_filters * 8, num_filters * 16, num_filters * 16)
        self.up3 = TernausUp(num_filters * 16, num_filters * 4, num_filters * 8, num_filters * 8)
        self.up2 = TernausUp(num_filters * 8, num_filters * 2, num_filters * 4, num_filters * 4)
        self.up1 = TernausUp(num_filters * 4, num_filters * 1, num_filters * 2, num_filters)
        self.outc = nn.Conv2d(num_filters, n_classes, kernel_size=1)
        self.up_features = [128, 256, 512, 512, 512]


class UNet16Layer4:
    def __init__(self, backbone, n_classes=1, num_filters=32):
        """
        :param backbone: vgg16
        :param n_classes:
        :param num_filters:
            False - no pre-trained network used
            True - encoder pre-trained with VGG16
        """
        super().__init__()
        if n_classes == 2:
            n_classes = 1

        self.n_classes = n_classes
        self.encoder = backbone.features

        self.inc = self.encoder[0:4]
        self.down1 = self.encoder[4:9]
        self.down2 = self.encoder[9:16]
        self.down3 = self.encoder[16:23]
        self.down4 = self.encoder[23:30]

        self.up4 = TernausUp(num_filters * 16, num_filters * 8, num_filters * 16, num_filters * 16)
        self.up3 = TernausUp(num_filters * 16, num_filters * 4, num_filters * 8, num_filters * 8)
        self.up2 = TernausUp(num_filters * 8, num_filters * 2, num_filters * 4, num_filters * 4)
        self.up1 = TernausUp(num_filters * 4, num_filters * 1, num_filters * 2, num_filters)
        self.outc = nn.Conv2d(num_filters, n_classes, kernel_size=1)
        self.up_features = [128, 256, 512, 512]


class UNet16BN:
    def __init__(self, backbone, n_classes=1, num_filters=32):
        """
        :param backbone: vgg16_bn
        :param n_classes:
        :param num_filters:
            False - no pre-trained network used
            True - encoder pre-trained with VGG16
        """
        super().__init__()
        if n_classes == 2:
            n_classes = 1

        self.n_classes = n_classes
        self.encoder = backbone.features

        self.inc = self.encoder[0:6]
        self.down1 = self.encoder[6:13]
        self.down2 = self.encoder[13:23]
        self.down3 = self.encoder[23:33]
        self.down4 = self.encoder[33:43]
        self.down5 = nn.Sequential(
            self.encoder[43],  # MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            ConvRelu(512, num_filters * 16)
        )

        self.up5 = TernausUp(num_filters * 16, num_filters * 8, num_filters * 16, num_filters * 16)
        self.up4 = TernausUp(num_filters * 16, num_filters * 8, num_filters * 16, num_filters * 16)
        self.up3 = TernausUp(num_filters * 16, num_filters * 4, num_filters * 8, num_filters * 8)
        self.up2 = TernausUp(num_filters * 8, num_filters * 2, num_filters * 4, num_filters * 4)
        self.up1 = TernausUp(num_filters * 4, num_filters * 1, num_filters * 2, num_filters)
        self.outc = nn.Conv2d(num_filters, n_classes, kernel_size=1)
        self.up_features = [128, 256, 512, 512, 512]


class UNet19:
    def __init__(self, backbone, n_classes=1, num_filters=32):
        """
        :param backbone: vgg16
        :param n_classes:
        :param num_filters:
        """
        super().__init__()
        if n_classes == 2:
            n_classes = 1

        self.n_classes = n_classes
        self.encoder = backbone.features

        self.inc = self.encoder[0:4]
        self.down1 = self.encoder[4:9]
        self.down2 = self.encoder[9:18]
        self.down3 = self.encoder[18:27]
        self.down4 = self.encoder[27:36]
        self.down5 = nn.Sequential(
            self.encoder[36],  # MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            ConvRelu(512, num_filters * 16)
        )

        self.up5 = TernausUp(num_filters * 16, num_filters * 8, num_filters * 16, num_filters * 16)
        self.up4 = TernausUp(num_filters * 16, num_filters * 8, num_filters * 16, num_filters * 16)
        self.up3 = TernausUp(num_filters * 16, num_filters * 4, num_filters * 8, num_filters * 8)
        self.up2 = TernausUp(num_filters * 8, num_filters * 2, num_filters * 4, num_filters * 4)
        self.up1 = TernausUp(num_filters * 4, num_filters * 1, num_filters * 2, num_filters)
        self.outc = nn.Conv2d(num_filters, n_classes, kernel_size=1)
        self.up_features = [128, 256, 512, 512, 512]


class UNet19Layer4:
    def __init__(self, backbone, n_classes=1, num_filters=32):
        """
        :param backbone: vgg19
        :param n_classes:
        :param num_filters:
        """
        super().__init__()
        if n_classes == 2:
            n_classes = 1

        self.n_classes = n_classes
        self.encoder = backbone.features

        self.inc = self.encoder[0:4]
        self.down1 = self.encoder[4:9]
        self.down2 = self.encoder[9:18]
        self.down3 = self.encoder[18:27]
        self.down4 = self.encoder[27:36]

        self.up4 = TernausUp(num_filters * 16, num_filters * 8, num_filters * 16, num_filters * 16)
        self.up3 = TernausUp(num_filters * 16, num_filters * 4, num_filters * 8, num_filters * 8)
        self.up2 = TernausUp(num_filters * 8, num_filters * 2, num_filters * 4, num_filters * 4)
        self.up1 = TernausUp(num_filters * 4, num_filters * 1, num_filters * 2, num_filters)
        self.outc = nn.Conv2d(num_filters, n_classes, kernel_size=1)
        self.up_features = [128, 256, 512, 512]


if __name__ == '__main__':
    model = vgg19()
    print(model)
