import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from lib.modeling.deeplab.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from lib.modeling.deeplab.aspp import build_aspp
from lib.modeling.deeplab.decoder import build_decoder
from lib.modeling.deeplab.backbone import build_backbone

sys.path.append('..')


class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone='resnet', output_stride=16,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if num_classes == 2:
            num_classes = 1
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        if self.training:
            return [x]
        else:
            return torch.sigmoid(x)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    model = DeepLab(backbone='mobilenet', output_stride=16, num_classes=1).cuda()
    model.eval()
    input = torch.rand(1, 3, 512, 512).cuda()
    output = model(input)
    print(output.size())
