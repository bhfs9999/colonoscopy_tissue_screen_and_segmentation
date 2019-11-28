from lib.modeling.unet.unet_frame import UNet
from lib.modeling.unet import dense_unet_121, res_unet_18, res_unet_34, \
    res_unet_50, res_unet_101, BaseUNet, unet11, unet16, unet19
from lib.modeling.dcan import DCAN
from lib.modeling.gcn import GCN
from lib.modeling.deeplab import DeepLab


def get_model(cfg, pre_train=True):
    model_name = cfg.MODEL.MODEL
    model_name = model_name.lower()

    if 'unet' in model_name:
        if model_name == 'unet':
            model = BaseUNet(n_classes=cfg.DATA.NUM_CLS)
        elif model_name == 'unet11':
            model = unet11(n_classes=cfg.DATA.NUM_CLS)
        elif model_name == 'unet11_no_pretrain':
            model = unet11(pre_train=False, n_classes=cfg.DATA.NUM_CLS)
        elif model_name == 'unet11se':
            model = unet11(se=True, n_classes=cfg.DATA.NUM_CLS)
        elif model_name == 'resunet11se':
            model = unet11(residual=True)
        elif model_name == 'unet16':
            model = unet16(n_classes=cfg.DATA.NUM_CLS)
        elif model_name == 'unet16bn':
            model = unet16(n_classes=cfg.DATA.NUM_CLS, bn=True)
        elif model_name == 'unet16layer4':
            model = unet16(pre_train=pre_train, n_classes=cfg.DATA.NUM_CLS, layer_4=True)
        elif model_name == 'unet19':
            model = unet19(n_classes=cfg.DATA.NUM_CLS)
        elif model_name == 'unet19layer4':
            model = unet19(pre_train=pre_train, n_classes=cfg.DATA.NUM_CLS, layer_4=True)
        elif model_name == 'denseunet121':
            model = dense_unet_121(n_classes=cfg.DATA.NUM_CLS)
        elif model_name == 'resunet18':
            model = res_unet_18(n_classes=cfg.DATA.NUM_CLS)
        elif model_name == 'resunet34':
            model = res_unet_34(n_classes=cfg.DATA.NUM_CLS)
        elif model_name == 'resunet50':
            model = res_unet_50(n_classes=cfg.DATA.NUM_CLS)
        elif model_name == 'resunet101':
            model = res_unet_101(n_classes=cfg.DATA.NUM_CLS)
        else:
            print('available model: unet, unet11, unet11_no_pretrain, unet11bn, '
                  'denseunet121, resunet(18, 34, 50, 101); dcan; gcn')
            raise ValueError
        model = UNet(model, deep_sup=cfg.MODEL.DEEP_SUP,
                     ds_se=cfg.MODEL.DEEP_SUP_SE, fusion_ds=cfg.MODEL.FUSION_DS,
                     cls_branch=cfg.MODEL.CLS_BRANCH)

    elif 'dcan' in model_name:
        model = DCAN()

    elif 'gcn' in model_name:
        model = GCN()

    elif 'deeplab' in model_name:
        if 'resnet' in model_name:
            backbone = 'resnet'
        elif 'xception' in model_name:
            backbone = 'xception'
        elif 'drn' in model_name:
            backbone = 'drn'
        elif 'mobilenet' in model_name:
            backbone = 'mobilenet'
        else:
            print('deeplab backbone only support resnet, xception, drc, mobilenet, get: ', model_name)
            raise ValueError
        model = DeepLab(num_classes=cfg.DATA.NUM_CLS, backbone=backbone)

    else:
        print('available model: unet, unet11, unet11_no_pretrain, unet11bn, '
              'denseunet121, resunet(18, 34, 50, 101); dcan; gcn; deeplab with backbone resnet, xception'
              'drc, mobilenet')
        raise ValueError

    return model


if __name__ == '__main__':
    from lib.loss.factory import LossWrapper, DiceLossV1
    from torch.nn import BCEWithLogitsLoss
    model = unet19(n_classes=1, layer_4=True)
    model = UNet(model, deep_sup='v2', cls_branch=True)
    import torch
    x = torch.randn(2, 3, 128, 128)
    target = (torch.randint(0, 2, [2, 1, 128, 128])).long()
    y = model(x)

    loss = LossWrapper(DiceLossV1(), BCEWithLogitsLoss())
    result = loss(target, y)
    y = y[0]
    print(y.shape)
