import torch
from torchvision import models
from .components import *
import torch.nn.functional as F
from lib.modeling.util import initialize_weights


class GCN(nn.Module):
    def __init__(self, input_size=512, num_classes=1, kernel_size=7):
        super().__init__()
        self.input_size = input_size
        num_classes = num_classes
        resnet = models.resnet152(pretrained=True)  # can change to resnet101/50

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.gcm1 = GlobalConvModule(2048, num_classes, (kernel_size, kernel_size))
        self.gcm2 = GlobalConvModule(1024, num_classes, (kernel_size, kernel_size))
        self.gcm3 = GlobalConvModule(512, num_classes, (kernel_size, kernel_size))
        self.gcm4 = GlobalConvModule(256, num_classes, (kernel_size, kernel_size))

        self.brm1 = BoundaryRefineModule(num_classes)
        self.brm2 = BoundaryRefineModule(num_classes)
        self.brm3 = BoundaryRefineModule(num_classes)
        self.brm4 = BoundaryRefineModule(num_classes)
        self.brm5 = BoundaryRefineModule(num_classes)
        self.brm6 = BoundaryRefineModule(num_classes)
        self.brm7 = BoundaryRefineModule(num_classes)
        self.brm8 = BoundaryRefineModule(num_classes)
        self.brm9 = BoundaryRefineModule(num_classes)

        initialize_weights(self.gcm1, self.gcm2, self.gcm3, self.gcm4, self.brm1, self.brm2, self.brm3,
                           self.brm4, self.brm5, self.brm6, self.brm7, self.brm8, self.brm9)

    # origin gcn model
    def forward(self, x, fea=False):
        feature = []
        # if x: 512
        fm0 = self.layer0(x)  # 256
        feature.append(fm0)
        fm1 = self.layer1(fm0)  # 128
        feature.append(fm1)
        fm2 = self.layer2(fm1)  # 64
        feature.append(fm2)
        fm3 = self.layer3(fm2)  # 32
        feature.append(fm3)
        fm4 = self.layer4(fm3)  # 16
        feature.append(fm4)

        gcfm1 = self.brm1(self.gcm1(fm4))  # 16
        feature.append(gcfm1)
        gcfm2 = self.brm2(self.gcm2(fm3))  # 32
        feature.append(gcfm2)
        gcfm3 = self.brm3(self.gcm3(fm2))  # 64
        feature.append(gcfm3)
        gcfm4 = self.brm4(self.gcm4(fm1))  # 128
        feature.append(gcfm4)

        fs1 = self.brm5(F.interpolate(gcfm1, fm3.size()[2:]) + gcfm2)  # 32
        feature.append(fs1)
        fs2 = self.brm6(F.interpolate(fs1, fm2.size()[2:]) + gcfm3)  # 64
        feature.append(fs2)
        fs3 = self.brm7(F.interpolate(fs2, fm1.size()[2:]) + gcfm4)  # 128
        feature.append(fs3)
        fs4 = self.brm8(F.interpolate(fs3, fm0.size()[2:]))  # 256
        feature.append(fs4)
        out = self.brm9(F.interpolate(fs4, self.input_size))  # 512
        feature.append(out)

        if self.training:
            return [out]
        else:
            return torch.sigmoid(out)
