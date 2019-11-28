from .loss import DiceLoss, BceLoss, CrossEntropyLoss, FocalLoss, DiceLossV1, BceWithLogDiceLoss
from torch import nn
from torch.nn import BCEWithLogitsLoss
import torch


def get_loss(cfg):
    """
    get loss from loss name, and warp it with a wrapper that handle deep supervision with multiple output
    :param cfg:
    :return:
    """
    loss_name = cfg.MODEL.LOSS
    loss_name = loss_name.lower()

    weight = cfg.MODEL.LOSS_WEIGHT
    if loss_name == 'diceloss':
        seg_loss = DiceLoss(weight=weight)
    elif loss_name == 'bceloss':
        seg_loss = BceLoss(weight=weight)
    elif loss_name == 'focalloss':
        seg_loss = FocalLoss(weight=weight, gamma=cfg.MODEL.LOSS_FL_GAMMA)
    elif loss_name == 'crossentropyloss':
        seg_loss = CrossEntropyLoss(weight=weight)
    elif loss_name == 'dicelossv1':
        seg_loss = DiceLossV1(weight=weight)
    elif loss_name == 'bcewithlogdiceloss':
        seg_loss = BceWithLogDiceLoss(class_weight=weight, bce_weight=cfg.MODEL.BCE_WEIGHT)
    else:
        print('loss must be DiceLoss, BceLoss, FocalLoss, DiceLossV1, CrossEntropyLoss, BceWithLogDiceLoss')
        raise ValueError
    cls_loss = BCEWithLogitsLoss()

    return LossWrapper(seg_loss, cls_loss, mirror_padding=cfg.DATA.MIRROR_PADDING)


class LossWrapper(nn.Module):
    """
    wrapper of loss, because
    1. model output is multilayer in deep supervision, need to calculate loss for each layer
    2. mirror padding in transformation, need to remove padding area before calculate loss
    """
    def __init__(self, seg_loss, cls_loss, mirror_padding=0, cls_weight=1):
        super(LossWrapper, self).__init__()
        self.seg_loss = seg_loss
        self.cls_loss = cls_loss
        self.cls_weight = cls_weight
        self.padding = mirror_padding

    def forward(self, gt, logits):
        """

        :param gt: tensor of shape B, C, H, W
        :param logits: tensor of shape B, C, H, W
        :return:
        """
        # remove padding
        if self.padding > 0:
            gt = gt[:, :, self.padding: -self.padding, self.padding: -self.padding]

        if isinstance(logits, list):
            if len(logits) == 2:
                # cls branch on
                seg_out, cls_out = logits

                # deep supervision with multiple output
                seg_loss = seg_out[0].new_zeros(1)
                count = 0.
                for seg_logit in seg_out:
                    if self.padding > 0:
                        seg_logit = seg_logit[:, :, self.padding: -self.padding, self.padding: -self.padding]
                    seg_loss += self.seg_loss(gt, seg_logit)
                    count += 1
                seg_loss /= count

                cls_target = gt.sum(dim=(1, 2, 3)) > 0  # B

                cls_loss = cls_out[0].new_zeros(1)
                for cls_logit in cls_out:
                    cls_loss += self.cls_loss(cls_logit, cls_target.float())
                    count += 1
                cls_loss /= count
                cls_weight = torch.tensor(self.cls_weight).to(cls_loss)

                print(float(cls_loss), float(seg_loss))
                if cls_loss == float("inf") or cls_loss == 0:
                    print(cls_out, [x.sum() for x in seg_out])
                loss_result = seg_loss + cls_weight * cls_loss

            elif len(logits) > 2:
                # deep supervision with multiple output
                loss_result = logits[0].new_zeros(1)
                count = 0.
                for logit in logits:
                    if self.padding > 0:
                        logit = logit[:, :, self.padding: -self.padding, self.padding: -self.padding]
                    loss_result += self.seg_loss(gt, logit)
                    count += 1
                loss_result /= count
            else:
                # no deep supervision, or fusion deep supervision
                logits = logits[0]
                if self.padding > 0:
                    logits = logits[:, :, self.padding: -self.padding, self.padding: -self.padding]
                loss_result = self.seg_loss(gt, logits)

        else:
            # inference loss
            if self.padding > 0:
                logits = logits[:, :, self.padding: -self.padding, self.padding: -self.padding]
            loss_result = self.seg_loss(gt, logits)

        return loss_result
