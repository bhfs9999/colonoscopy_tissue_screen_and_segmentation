import torch.nn as nn


class ConvBlock(nn.Sequential):
    """
    The implementation of Convolution block.
    """

    def __init__(self, inchannel, outchannel, kernelsize=3, padding=1, bias=False):
        """

        :param inchannel: The input channels of convolution layer.
        :param outchannel: The output channels of convolution layer.
        :param kernelsize: The kernel size of convolution layer.
        :param padding: Padding.
        :param bias: Bias.
        """

        super(ConvBlock, self).__init__()

        if bias == False:
            self.add_module('conv', nn.Sequential(
                nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=kernelsize, padding=padding, bias=False),
                nn.BatchNorm2d(outchannel),
                nn.ReLU()
            ))
        else:
            self.add_module('conv', nn.Sequential(
                nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=kernelsize, padding=padding),
                nn.ReLU()
            ))


class AuxiliaryBlock(nn.Sequential):
    """
    The implementation of Auxiliary block.
    """

    def __init__(self, inchannel, k, padding):
        """

        :param inchannel: The input channels of up-convolution layer.
        :param k: The hyper-parameter of up-convolution layer.
        :param padding: Padding of up-convolution layer.
        """
        super(AuxiliaryBlock, self).__init__(
            nn.ConvTranspose2d(in_channels=inchannel, out_channels=2, kernel_size=2 * k, stride=k, padding=padding),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)
        )
