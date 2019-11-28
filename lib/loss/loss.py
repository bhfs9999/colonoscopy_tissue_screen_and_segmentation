from torch import nn
import torch
from torch.nn import functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-7, weight=None):
        """
        Diceloss for segmentation
        :param smooth: smooth value for fraction
        :param weight: class weight
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        if weight is not None:
            weight = torch.tensor(weight).float()
        self.weight = weight

    def forward(self, gt, logits, reduction='mean'):
        """
        code from https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
        Note that PyTorch optimizers minimize a loss. In this case,
        we would like to maximize the dice loss so we return the negated dice loss.
        :param gt: a tensor of shape [B, 1 , H, W]
        :param logits: a tensor of shape [B, C, H, W]. Corresponds to the raw output or logit of the model.
        :param reduction: 'none' to return dice coff of each image of each batch, 'mean' to return averaged dice loss
        :return:
            dice_loss: dice loss
        """
        num_classes = logits.shape[1]
        gt = gt.long()
        if num_classes == 1:
            gt_1_hot = torch.eye(num_classes + 1)[gt.squeeze(1)].to(gt.device)  # B,H,W,2
            gt_1_hot = gt_1_hot.permute(0, 3, 1, 2).contiguous().float()  # B,2,H,W
            pos_prob = torch.sigmoid(logits)  # B,1,H,W
            neg_prob = 1 - pos_prob  # B,1,H,W
            probas = torch.cat([neg_prob, pos_prob], dim=1)  # B,2,H,W
        else:
            gt_1_hot = torch.eye(num_classes)[gt.squeeze(1)]  # B,H,W,cls
            gt_1_hot = gt_1_hot.permute(0, 3, 1, 2).float()  # B,cls,H,W
            probas = F.softmax(logits, dim=1)  # B,cls,H,W

        # gt_1_hot = gt_1_hot.type(logits.type())
        # two different implementation
        # sum of inter among batch / sum of cardinality among batch
        if False:
            dims = (0, ) + tuple(range(2, logits.ndimension()))  # (0, 2, 3), sum on batch, h, w
            intersection = torch.sum(probas * gt_1_hot, dims)  # len: cls
            cardinality = torch.sum(probas + gt_1_hot, dims)  # len: cls
            dice_coff = (2 * intersection / (cardinality + self.smooth))
            dice_loss = 1.0 - dice_coff
            if self.weight is not None:
                weight = self.weight.to(logits.device)
                dice_loss = weight * dice_loss
            dice_loss = dice_loss.mean()

        # sum of (inter / cardinality) of each image, then average
        else:
            dims = tuple(range(2, logits.ndimension()))
            intersection = torch.sum(probas * gt_1_hot, dims)  # B,cls
            cardinality = torch.sum(probas + gt_1_hot, dims)  # B,cls
            dice_coff = 2 * intersection / (cardinality + self.smooth)  # B, cls
            if reduction == 'none':
                return dice_coff
            dice_loss = 1.0 - dice_coff  # B, cls
            if self.weight is not None:
                weight = self.weight.to(logits.device)
                dice_loss = weight * dice_loss  # B, cls
            if reduction == 'mean':
                dice_loss = dice_loss.mean()  # 1

        return dice_loss


class DiceLossV1(nn.Module):
    """
    different from normal dice loss
    for image with no positive mask, only calculate dice for background
    for positive image, only calculate positive pixel
    """
    def __init__(self, smooth=1e-7, weight=None):
        """
        Diceloss for segmentation
        :param smooth: smooth value for fraction
        :param weight: class weight
        """
        super(DiceLossV1, self).__init__()
        self.smooth = smooth
        if weight is not None:
            weight = torch.tensor(weight).float()
        self.weight = weight

    def forward(self, gt, logits, reduction='mean'):
        """
        code from https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
        Note that PyTorch optimizers minimize a loss. In this case,
        we would like to maximize the dice loss so we return the negated dice loss.
        :param gt: a tensor of shape [B, 1 , H, W]
        :param logits: a tensor of shape [B, C, H, W]. Corresponds to the raw output or logit of the model.
        :param reduction:
        :return:
            dice_loss: dice loss
        """
        num_classes = logits.shape[1]
        gt = gt.long()
        if num_classes == 1:
            gt_1_hot = torch.eye(num_classes + 1)[gt.squeeze(1)]  # B,H,W,2
            gt_1_hot = gt_1_hot.permute(0, 3, 1, 2).float()  # B,2,H,W
            gt_1_hot = torch.cat([gt_1_hot[:, 0:1, :, :],
                                  gt_1_hot[:, 1:2, :, :]],
                                 dim=1)
            pos_prob = torch.sigmoid(logits)  # B,1,H,W
            neg_prob = 1 - pos_prob  # B,1,H,W
            probas = torch.cat([neg_prob, pos_prob], dim=1)  # B,2,H,W
        else:
            gt_1_hot = torch.eye(num_classes)[gt.squeeze(1)]  # B,H,W,cls
            gt_1_hot = gt_1_hot.permute(0, 3, 1, 2).float()  # B,cls,H,W
            probas = F.softmax(logits, dim=1)  # B,cls,H,W

        gt_1_hot = gt_1_hot.type(logits.type())

        # whether gt have pos pixel
        is_pos = gt.gt(0).sum(dim=(1, 2, 3)).gt(0)  # B

        dims = tuple(range(2, logits.ndimension()))
        intersection = torch.sum(probas * gt_1_hot, dims)  # B,cls
        cardinality = torch.sum(probas + gt_1_hot, dims)  # B,cls
        dice_coff = 2 * intersection / (cardinality + self.smooth)  # B,cls
        if reduction == 'none':
            dice_coff = torch.where(is_pos, dice_coff[:, 1:].mean(1), dice_coff[:, 0])
            return dice_coff
        dice_loss = 1 - dice_coff
        if self.weight is not None:
            weight = self.weight.to(logits.device)  # cls
            dice_loss = weight * dice_loss  # B, cls

        dice_loss = (torch.where(is_pos, dice_loss[:, 1:].mean(1), dice_loss[:, 0])).mean()

        return dice_loss


class BceLoss(nn.Module):
    def __init__(self, weight=None):
        """
        Computes the weighted binary cross-entropy loss.
        :param weight: a scalar representing the weight attributed
                        to the positive class. This is especially useful for
                        an imbalanced dataset.
        """
        super(BceLoss, self).__init__()
        if weight is not None:
            if isinstance(weight, (list, tuple)) and len(weight) == 2:
                weight = weight[1]
            weight = torch.tensor(weight).float()
        self.weight = weight

    def forward(self, gt, logits, reduction='mean'):
        """
        Computes the weighted binary cross-entropy loss.
        :param gt: a tensor of shape [B, 1, H, W].
        :param logits: a tensor of shape [B, 1, H, W]. Corresponds to
        :param reduction: same as F.binary_cross_entropy_with_logits
        :return: bce_loss: the weighted binary cross-entropy loss.
        """
        num_class = logits.shape[1]
        assert num_class <= 2, "For class num larger than 2, use CrossEntropy instead"
        if self.weight is not None:
            weight = self.weight.to(logits.device)
        else:
            weight = None
        if num_class == 2:
            gt = torch.eye(2)[gt.squeeze(1).long()].to(logits.device)  # B, H, W, 2
            gt = gt.permute(0, 3, 1, 2).contiguous()  # B, 2, H, W

        bce_loss = F.binary_cross_entropy_with_logits(
            logits.float(),
            gt.float(),
            reduction=reduction,
            pos_weight=weight
        )

        return bce_loss


class BceWithLogDiceLoss(nn.Module):
    """
    bce loss - ln(dice) for segmentation
    :param smooth: smooth value for fraction
    :param class_weight: class weight for both bce loss and dice loss
    :param bce_weight: result is bce_loss * bce_weight + dice_loss
    """
    def __init__(self, smooth=1e-7, class_weight=None, bce_weight=1.):
        super(BceWithLogDiceLoss, self).__init__()

        self.bce_weight = (torch.tensor([bce_weight]) / (bce_weight + 1)).float()
        self.dice_weight = (torch.tensor([1.]) / (bce_weight + 1)).float()
        self.dice_loss = DiceLossV1(smooth=smooth, weight=class_weight)
        self.bce_loss = BceLoss(weight=class_weight)

    def forward(self, gt, logits):
        bce_loss = self.bce_loss(gt, logits, reduction='none')  # b, c, w, h
        dice_coff = self.dice_loss(gt, logits,  reduction='none')  # b

        bce_loss = bce_loss.mean(dim=(1, 2, 3))  # b
        dice_loss = - torch.log(dice_coff)  # ln dice loss = - ln(dice)

        loss = bce_loss * self.bce_weight.to(bce_loss.device) + dice_loss * self.dice_weight.to(bce_loss.device)
        loss = loss.mean()

        return loss


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        """
        computes the weighted multi-class cross-entropy loss.
        :param weight:a tensor of shape [C,], tuple or list.
        The weights attributed to each class.
        """
        super(CrossEntropyLoss, self).__init__()
        if weight is not None:
            weight = torch.tensor(weight).float()
        self.weight = weight

    def forward(self, gt, logits):
        """
        computes the weighted multi-class cross-entropy loss.
        :param gt: a tensor of shape [B, 1, H, W].
        :param logits: a tensor of shape [B, C, H, W]. Corresponds to the raw output or logits of the model.
        :return: ce_loss: the weighted multi-class cross-entropy loss.
        """
        num_class = logits.shape[1]
        assert num_class > 2, "For class num 1 or 2, use BecLoss instead"
        if self.weight is not None:
            weight = self.weight.to(logits.device)
        else:
            weight = None
        ce_loss = F.cross_entropy(
            logits.float(),
            gt.long().squeeze(1),
            weight=weight
        )

        return ce_loss


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        """
        computes focal loss
        :param alpha: focal loss cofficient
        :param gamma: focal loss cofficient
        """
        super(FocalLoss, self).__init__()
        if weight is not None:
            weight = torch.tensor(weight).float()
        self.weight = weight
        self.gamma = gamma

    def forward(self, gt, logits):
        """
        computes focal loss
        :param gt:  a tensor of shape [B, 1, H, W].
        :param logits:  a tensor of shape [B, C, H, W]. Corresponds to the raw output or logits of the model.
        :return:
        """
        num_classes = logits.shape[1]
        gt = gt.long()
        if num_classes == 1:
            num_classes = 2
            gt_1_hot = torch.eye(num_classes)[gt.squeeze(1)]  # B,H,W,2
            gt_1_hot = gt_1_hot.float()  # B,H,W,2

            pos_prob = torch.sigmoid(logits)  # B,1,H,W
            neg_prob = 1 - pos_prob  # B,1,H,W
            probas = torch.cat([neg_prob, pos_prob], dim=1)  # B,2,H,W
            probas = probas.permute(0, 2, 3, 1).contiguous()  # B,H,W,2
        else:
            gt_1_hot = torch.eye(num_classes)[gt.squeeze(1)]  # B,H,W,cls
            gt_1_hot = gt_1_hot.float()  # B,H,W,cls

            probas = F.softmax(logits, dim=1)  # B,cls,H,W
            probas = probas.permute(0, 2, 3, 1).contiguous()  # B,H,W,cls

        gt_1_hot = gt_1_hot.type(logits.type())
        gt_1_hot = gt_1_hot.view(-1, num_classes)  # (B*H*W), cls
        probas = probas.view(-1, num_classes)  # (B*H*W), cls

        # focal loss: - (1 - gt*probas)^gamma * log(gt*probas)
        raw_loss = (gt_1_hot * probas).sum(1)  # (B*H*W)
        log_loss = raw_loss.log()
        focal = torch.pow((1 - raw_loss), self.gamma)
        focal_loss = - focal * log_loss  # (B*H*W)
        if self.weight is not None:
            device = focal_loss.device
            weight = self.weight.to(device)
            weight = weight[gt].permute(0, 2, 3, 1).contiguous().view(-1, 1).squeeze(1)  # (B*H*W)
            focal_loss = focal_loss * weight

        return focal_loss.mean()


if __name__ == '__main__':
    rand_logits = torch.rand(8, 1, 512, 512).float()
    rand_gts = (torch.randint(0, 2, [8, 1, 512, 512])).long()
    loss = BceWithLogDiceLoss(class_weight=(1, 1))
    fl = loss(rand_gts, rand_logits)
    print(fl)
