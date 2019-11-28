import torch
from .components import *
import torch.nn.functional as F

from lib.modeling.util import initialize_weights


class DCAN(nn.Module):
    """
    The implementation of Deep Contextual Network
    """

    def __init__(self):
        super(DCAN, self).__init__()

        #Main path
        self.conv1 = ConvBlock(3, 64)
        self.conv2 = ConvBlock(64, 64)

        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 128)

        self.conv5 = ConvBlock(128, 256)
        self.conv6 = ConvBlock(256, 256)
        self.conv7 = ConvBlock(256, 256)

        self.conv8 = ConvBlock(256, 512)
        self.conv9 = ConvBlock(512, 512)
        self.conv10 = ConvBlock(512, 512)

        #Auxiliary path
        self.auxi1 = AuxiliaryBlock(128, 2, 1)
        self.auxi2 = AuxiliaryBlock(256, 4, 2)
        self.auxi3 = AuxiliaryBlock(512, 8, 4)

        initialize_weights(self)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv3(x)
        x = self.conv4(x)
        out1 = self.auxi1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        out2 = self.auxi2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        out3 = self.auxi3(x)

        final_out = out1 + out2 + out3
        self.aux_out = out1, out2, out3
        if self.training:
            return [final_out]
        else:
            return torch.sigmoid(final_out)
