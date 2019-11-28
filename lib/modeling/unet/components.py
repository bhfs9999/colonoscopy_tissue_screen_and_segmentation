import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.modeling.components import ResBlock, SELayer
from lib.modeling.util import initialize_weights


class SingleConv(nn.Module):
    """(BN => ReLU => conv => BN => Relu) * 1 for dense unet"""

    def __init__(self, in_ch, out_ch):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        initialize_weights(self.conv)

    def forward(self, x):
        y = self.conv(x)
        return y


class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        initialize_weights(self.conv)

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)
        initialize_weights(self.up)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        initialize_weights(self.conv)

    def forward(self, x):
        x = self.conv(x)
        return x


class DeepSupervisionBlockV1(nn.Module):
    def __init__(self, in_ch, num_cls, up_scale, se_module=False):
        super(DeepSupervisionBlockV1, self).__init__()
        self.up = nn.Sequential(
            ResBlock(in_ch, se_module=se_module),
            nn.Conv2d(in_ch, num_cls, kernel_size=(1, 1)),
            nn.Upsample(scale_factor=up_scale, mode='bilinear', align_corners=True),
        )
        initialize_weights(self.up)

    def forward(self, x):
        x = self.up(x)
        return x


class DeepSupervisionBlockV2(nn.Module):
    def __init__(self, in_ch, num_cls, up_scale, se_module=False):
        super(DeepSupervisionBlockV2, self).__init__()
        self.up = nn.Sequential(
            ResBlock(in_ch, se_module=se_module),
            nn.Conv2d(in_ch, num_cls, kernel_size=(1, 1)),
            nn.Upsample(scale_factor=up_scale, mode='bilinear', align_corners=True),
            nn.Conv2d(num_cls, num_cls, kernel_size=1)
        )
        initialize_weights(self.up)

    def forward(self, x):
        x = self.up(x)
        return x


class DeepSupervisionFusionBlockV1(nn.Module):
    """
    conv each deep supervision output with different size, which becomes larger when output is from deeper ds path,
    then add then and use a 1*1 conv to get final result
    """
    def __init__(self, cls_num=1, layer_nums=4):
        super(DeepSupervisionFusionBlockV1, self).__init__()
        self.ds_paths = nn.ModuleList([
            nn.Conv2d(cls_num, cls_num, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(cls_num, cls_num, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(cls_num, cls_num, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(cls_num, cls_num, kernel_size=7, stride=1, padding=3),
            nn.Conv2d(cls_num, cls_num, kernel_size=9, stride=1, padding=4)
        ])

        if layer_nums == 5:
            self.ds_paths.append(nn.Conv2d(cls_num, cls_num, kernel_size=11, stride=1, padding=5))

        self.out_conv = nn.Conv2d(cls_num, cls_num, kernel_size=1)

        initialize_weights(self)

    def forward(self, out):
        ds_out = []
        for i, outi in enumerate(out):
            ds_out.append([self.ds_paths[i](outi)])

        final_out = self.out_conv(torch.sum(ds_out))  # b, 1, h, w

        return final_out


class DeepSupervisionFusionBlockV2(nn.Module):
    """
    instead add ds path's result nin v1, v2 concatenate them, then feed to a 1*1 conv
    """
    def __init__(self, cls_num=1, layer_nums=4):
        super(DeepSupervisionFusionBlockV2, self).__init__()
        self.ds_paths = nn.ModuleList([
            nn.Conv2d(cls_num, cls_num, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(cls_num, cls_num, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(cls_num, cls_num, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(cls_num, cls_num, kernel_size=7, stride=1, padding=3),
            nn.Conv2d(cls_num, cls_num, kernel_size=9, stride=1, padding=4)
        ])

        if layer_nums == 5:
            self.ds_paths.append(nn.Conv2d(cls_num, cls_num, kernel_size=11, stride=1, padding=5))

        self.out_conv = nn.Conv2d(cls_num * (layer_nums + 1), cls_num, kernel_size=1)

        initialize_weights(self)

    def forward(self, out):
        ds_out = []
        for i, outi in enumerate(out):
            ds_out.append(self.ds_paths[i](outi))

        final_out = self.out_conv(torch.cat(ds_out, 1))  # b, 1, h, w

        return final_out


class DeepSupervisionFusionBlockV3(nn.Module):
    """
    concatenate 5 deep supervision output, then use a 1*1 conv to get final output
    """
    def __init__(self, cls_num=1, layer_nums=4):
        super(DeepSupervisionFusionBlockV3, self).__init__()
        self.out_conv = nn.Conv2d((layer_nums + 1) * cls_num, cls_num, kernel_size=1)

        initialize_weights(self)

    def forward(self, out):
        final_out = self.out_conv(torch.cat(out, 1))  # b, 1, h, w

        return final_out


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class ConvBNRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = nn.Sequential(conv3x3(in_, out),
                                  nn.BatchNorm2d(out))
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class TernausUp(nn.Module):
    """
    TernausNet upsample model
    """
    def __init__(self, in_channels, in_up_channels, skip_conn_channels, out_channels, bn=False):
        super().__init__()
        if bn:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_up_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(in_up_channels),
                nn.ReLU(inplace=True))
            self.conv = ConvBNRelu(in_up_channels + skip_conn_channels, out_channels)
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_up_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True))
            self.conv = ConvRelu(in_up_channels + skip_conn_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        y = self.conv(torch.cat([x1, x2], 1))

        return y


class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                Interpolate(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor,
                        mode=self.mode, align_corners=self.align_corners)
        return x


class ResidualEncoder(nn.Module):
    def __init__(self, encoder, channels, se_block=False):
        super(ResidualEncoder, self).__init__()
        self.encoder = encoder
        self.se_block = se_block
        if se_block:
            self.se = SELayer(channels=channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.encoder(x)
        if self.se_block:
            x = self.se(x)
        x += identity
        y = self.relu(x)
        return y


class ClsBranch(nn.Module):
    def __init__(self, in_ch, se_module=False, num_cls=1):
        super(ClsBranch, self).__init__()
        self.cls_branch = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResBlock(in_ch=in_ch, se_module=se_module),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResBlock(in_ch=in_ch, se_module=se_module),
            nn.Conv2d(in_ch, num_cls, kernel_size=1),
            nn.AdaptiveMaxPool2d(1)
        )

        initialize_weights(self)

    def forward(self, x):
        y = self.cls_branch(x)

        return y


if __name__ == '__main__':
    model = DeepSupervisionBlockV2()
