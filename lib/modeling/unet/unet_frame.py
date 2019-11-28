from lib.modeling.unet.vgg_unet import *


class UNet(nn.Module):
    def __init__(self, backbone, deep_sup='', ds_se=False, fusion_ds='', layer_nums=4, cls_branch=False):
        """
        unet with 4 down sample and 4 up sample
        :param backbone: variance model backbone
        :param deep_sup: whether use deep supervision, '' for not use, v1 use upsample result, v2 add 1*1 conv after
        upsample result
        :param ds_se: use se block in ds path
        :param fusion_ds: whether to fuse results from multi-layer deep supervision, use only when deep_sup=True,
        must be v1 or v2 or '', v1 add, v2 concatenate, '' no fusion
        :param layer_nums: downsample and up sample times, for original unet, there are 4 downsample and 4 upsample,
        :param cls_branch: whether add cls branch for multi-task learning, use with deep_sup
        for ternausnet, this para is 5
        """
        super(UNet, self).__init__()
        assert layer_nums in [4, 5], 'layer nums is 4 for Unet, 5 for TernausNet'
        assert not cls_branch or (deep_sup != '' and cls_branch), 'classification branch must use with deep supervision'

        self.inc = backbone.inc
        self.encoder1 = backbone.down1
        self.encoder2 = backbone.down2
        self.encoder3 = backbone.down3
        self.encoder4 = backbone.down4

        self.decoder1 = backbone.up1
        self.decoder2 = backbone.up2
        self.decoder3 = backbone.up3
        self.decoder4 = backbone.up4
        self.layer_nums = layer_nums
        if layer_nums == 5:
            self.encoder5 = backbone.down5
            self.decoder5 = backbone.up5
        self.out = backbone.outc

        self.deep_sup = deep_sup
        self.cls = cls_branch
        up_features = backbone.up_features
        n_classes = backbone.n_classes
        self._build_ds_path(deep_sup, fusion_ds, ds_se, up_features, n_classes, layer_nums, cls_branch)

    def _build_ds_path(self, deep_sup, fusion_ds, ds_se, up_features, n_classes, layer_nums=4, cls_branch=False):
        """
        build deep supervision blocks for model
        :param deep_sup: whether use deep supervision, '' for not use, v1 use upsample result, v2 add 1*1 conv after
        upsample result
        :param fusion_ds: whether to fuse results from multi-layer deep supervision, use only when deep_sup=True,
        must be v1 or v2 or '', v1 add, v2 concatenate, '' no fusion
        :param ds_se: use se block in ds path
        :param up_features: right path's channels
        :param n_classes:
        :param layer_nums:
        :param cls: whether add cls branch for multi-task learning, use with deep_sup
        :return:
        """
        self.deep_sup = deep_sup
        if self.deep_sup != '':
            if self.deep_sup == 'v1':
                self.ds_out = nn.ModuleList([
                    DeepSupervisionBlockV1(up_features[0], n_classes, 2, se_module=ds_se),
                    DeepSupervisionBlockV1(up_features[1], n_classes, 4, se_module=ds_se),
                    DeepSupervisionBlockV1(up_features[2], n_classes, 8, se_module=ds_se),
                    DeepSupervisionBlockV1(up_features[3], n_classes, 16, se_module=ds_se)
                ])
                if layer_nums == 5:
                    self.ds_out.append(DeepSupervisionBlockV1(up_features[4], n_classes, 32, se_module=ds_se))
            elif self.deep_sup == 'v2':
                self.ds_out = nn.ModuleList([
                    DeepSupervisionBlockV2(up_features[0], n_classes, 2, se_module=ds_se),
                    DeepSupervisionBlockV2(up_features[1], n_classes, 4, se_module=ds_se),
                    DeepSupervisionBlockV2(up_features[2], n_classes, 8, se_module=ds_se),
                    DeepSupervisionBlockV2(up_features[3], n_classes, 16, se_module=ds_se)
                ])
                if layer_nums == 5:
                    self.ds_out.append(DeepSupervisionBlockV2(up_features[4], n_classes, 32, se_module=ds_se))
            else:
                print("deep supervision method error, get {}, but expect 'v1', 'v2' or''".format(fusion_ds))
                raise ValueError
            self.fusion_ds = fusion_ds
            if fusion_ds == 'v1':
                self.fusion_block = DeepSupervisionFusionBlockV1(cls_num=n_classes, layer_nums=layer_nums)
            elif fusion_ds == 'v2':
                self.fusion_block = DeepSupervisionFusionBlockV2(cls_num=n_classes, layer_nums=layer_nums)
            elif fusion_ds == 'v3':
                self.fusion_block = DeepSupervisionFusionBlockV3(cls_num=n_classes, layer_nums=layer_nums)
            elif fusion_ds != '':
                print("deep supervision fusion method error, get {}, "
                      "but expect 'v1', 'v2', 'v3' or''".format(fusion_ds))
                raise ValueError

            if cls_branch:
                self.cls_branch = nn.ModuleList([
                    ClsBranch(32, se_module=True),
                    ClsBranch(up_features[0], se_module=True, num_cls=n_classes),
                    ClsBranch(up_features[1], se_module=True, num_cls=n_classes),
                    ClsBranch(up_features[2], se_module=True, num_cls=n_classes),
                    ClsBranch(up_features[3], se_module=True, num_cls=n_classes)
                ])
                if layer_nums == 5:
                    self.cls_branch.append((ClsBranch(up_features[4], se_module=True, num_cls=n_classes)))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)
        if self.layer_nums == 5:
            x6 = self.encoder5(x5)
            y5 = self.decoder5(x6, x5)
            y4 = self.decoder4(y5, x4)
        else:
            y4 = self.decoder4(x5, x4)
        y3 = self.decoder3(y4, x3)
        y2 = self.decoder2(y3, x2)
        y1 = self.decoder1(y2, x1)
        out_final = self.out(y1)

        # input for deep supervision block
        if self.layer_nums == 5:
            ds_input = [y2, y3, y4, y5, x6]
            cls_input = [y1, y2, y3, y4, y5, x6]
        else:
            ds_input = [y2, y3, y4, x5]
            cls_input = [y1, y2, y3, y4, x5]

        seg_out = [out_final]
        cls_out = []
        if self.deep_sup:
            for i in range(len(ds_input)):
                seg_out.append(self.ds_out[i](ds_input[i]))

        if self.deep_sup and self.fusion_ds:
            seg_out = self.fusion_block(seg_out)
            seg_out = [seg_out]

        if self.cls:
            cls_out = [m(x).squeeze() for x, m in zip(cls_input, self.cls_branch)]

        if self.training:
            if self.cls:
                return [seg_out, cls_out]
            else:
                return seg_out
        else:
            return torch.sigmoid(seg_out[0])


if __name__ == '__main__':
    # model = UNet(unet16(), deep_sup='v2', ds_se=True, fusion_ds='v2', layer_nums=5)
    # x = torch.ones(2, 3, 128, 128)
    # y = model(x)
    # y = y[0]
    # print(y.shape)
    model = vgg11()
    print(model)



# if __name__ == '__main__':
#     model = UNet(BaseUNet(), deep_sup='', ds_se=True, fusion_ds='v2').cuda()
#     x = torch.ones(2, 3, 128, 128, requires_grad=True).cuda()
#     gt = torch.ones(2, 1, 128, 128, requires_grad=True).cuda()
#
#     from torch.optim import SGD
#     optim = SGD(model.parameters(), lr=0.01, momentum=0.99)
#     from torch.nn.functional import binary_cross_entropy_with_logits
#     from torch.nn.utils import clip_grad_norm_
#
#     while True:
#         dt = model(x)
#         dt = dt[0]
#         loss = binary_cross_entropy_with_logits(dt, gt)
#
#         optim.zero_grad()
#
#         loss.backward()
#         norm = clip_grad_norm_(model.parameters(), max_norm=1)
#         print(loss, norm)
#         optim.step()

# tensor(1.1233, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 4.513642169908741
# tensor(1.0815, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 4.092784145654968
# tensor(1.0174, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 3.650201592859796
# tensor(0.9486, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 3.4252044698329516
# tensor(0.8782, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 3.1519412262777924
# tensor(0.8058, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 2.9124495299213438
# tensor(0.7317, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 2.5440250517381653
# tensor(0.6591, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 2.258024563867602
# tensor(0.5883, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 2.116685240950862
# tensor(0.5179, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 1.7911445096814786
# tensor(0.4494, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 1.564174109718984
# tensor(0.3837, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 1.3583079038906596
# tensor(0.3218, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 1.0960053482933099
# tensor(0.2657, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.9049812624383544
# tensor(0.2164, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.7560424653467585
# tensor(0.1740, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.6230324308064983
# tensor(0.1382, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.5080398158818682
# tensor(0.1088, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.4098503999836622
# tensor(0.0853, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.3332038215569166
# tensor(0.0666, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.26977639695152666
# tensor(0.0519, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.21779206197674433
# tensor(0.0405, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.17600310977733544
# tensor(0.0315, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.1430478057502367
# tensor(0.0245, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.11617714778622194
# tensor(0.0190, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.09323331927376612
# tensor(0.0148, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.0750991275806659
# tensor(0.0115, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.05994625348939027
# tensor(0.0090, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.048074459270806995
# tensor(0.0070, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.03848501786591702
# tensor(0.0055, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.030888170589072173
# tensor(0.0043, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.02512450898113587
# tensor(0.0034, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.020511301245469536
# tensor(0.0027, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.016783946225600803
# tensor(0.0021, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.013730740393442952
# tensor(0.0017, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.011231126036910021
# tensor(0.0013, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.00921255914498892
# tensor(0.0010, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.007560576352972058
# tensor(0.0008, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.006219946088268539
# tensor(0.0007, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.005135321316925269
# tensor(0.0005, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.004240733194603738
# tensor(0.0004, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.0035026049190844677
# tensor(0.0003, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.00289494318850419
# tensor(0.0003, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.002403796083834763
# tensor(0.0002, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.0020071379000586543
# tensor(0.0002, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.0016724640790889065
# tensor(0.0001, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.0013976885883327799
# tensor(0.0001, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>) 0.0011715333664464736
