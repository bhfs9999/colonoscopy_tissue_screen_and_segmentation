from collections import defaultdict
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import numpy as np
import torch


def auc_score(gt_mask, pred_mask):
    targets = gt_mask.view(-1).long()
    scores = pred_mask.view(-1).float()

    scores, sortind = scores.sort(dim=0, descending=True)
    tpr = torch.zeros(scores.shape[0] + 1).float()
    fpr = torch.zeros(scores.shape[0] + 1).float()

    for i in range(1, scores.shape[0] + 1):
        if targets[sortind[i - 1]] == 1:
            tpr[i] = tpr[i - 1] + 1
            fpr[i] = fpr[i - 1]
        else:
            tpr[i] = tpr[i - 1]
            fpr[i] = fpr[i - 1] + 1

    tpr /= (targets.sum() * 1.0)
    fpr /= ((targets - 1.0).sum() * -1.0)

    n = tpr.shape[0]
    h = fpr[1: n] - fpr[0: n - 1]
    sum_h = torch.zeros(fpr.shape)
    sum_h[0: n - 1] = h
    sum_h[1: n] += h
    area = (sum_h * tpr).sum() / 2.0

    return area, tpr, fpr


def dice_score(gt_mask, pred_mask, smooth=1e-7):
    """
    calculate dice score of prediction
    :param gt_mask:  ndarray of shape [H, W]
    :param pred_mask:  ndarray of shape [C, H, W]
    :param smooth: smooth value
    :return: dice score of each class, average dice
    """
    gt_mask = gt_mask.to(torch.uint8)
    device = gt_mask.device

    num_classes = pred_mask.shape[0]
    if num_classes == 1:
        num_classes = 2
        pred_1_hot = torch.eye(2, dtype=torch.uint8)[pred_mask.squeeze(0).long()].to(device)  # H,W,2
    else:
        pred_mask = torch.argmax(pred_mask, dim=0).to(torch.uint8)  # 1,H,W
        pred_1_hot = torch.eye(num_classes, dtype=torch.uint8)[pred_mask.long()].to(device)  # H,W,C

    gt_1_hot = torch.eye(num_classes, dtype=torch.uint8)[gt_mask.long()].to(device)  # H,W,C

    intersection = torch.sum(gt_1_hot * pred_1_hot, dim=(0, 1)).float()  # cls
    cardinality = torch.sum(gt_1_hot + pred_1_hot, dim=(0, 1)).float()  # cls
    dice_coff = 2. * intersection / (cardinality + smooth)  # cls

    return dice_coff, torch.mean(dice_coff)


def get_cls_label(pred_mask, th_ratio=0.003, th_pixel=30000):
    pred_area = pred_mask.sum()
    total_area = pred_mask.shape[-1] * pred_mask.shape[-2]
    ratio = float(pred_area) / total_area
    return int(ratio >= th_ratio and pred_area >= th_pixel)


def do_evaluation(gt_masks, pred_masks, detail_info=False):
    """
    evaluate on model result, calculate dice,
    :param gt_masks: list of ndarray of shape [1, H, W]
    :param pred_masks: list of ndarray of shape [C, H, W]
    :param detail_info: whether return each image's metric result
    :param need_auc: whether calculate auc, auc calculation is too slow
    :return:
    """
    result_list = defaultdict(list)

    for gt_mask, pred_mask in zip(gt_masks, pred_masks):
        if len(gt_mask.shape) == 3:
            gt_mask = gt_mask.squeeze(0)

        if pred_mask.dtype != np.uint8:
            # merge method is prob_mean, pred_mask is of type float32
            pred_mask[pred_mask >= 0.5] = 1
            pred_mask[pred_mask < 0.5] = 0
            pred_mask.astype(np.uint8)

        # classification
        gt_label = int(gt_mask.sum() > 0)
        pred_label = get_cls_label(pred_mask)
        result_list['gt_label'].append(gt_label)
        result_list['pred_label'].append(pred_label)

        # segmentation
        if gt_label == 1:
            # positive sample, calculate dice
            gt_mask = torch.from_numpy(gt_mask)
            pred_mask = torch.from_numpy(pred_mask)

            dices, dice_avg = dice_score(gt_mask, pred_mask)

            result_list['dice_1'].append(float(dices[1].cpu()))
        else:
            result_list['dice_1'].append(-1.0)

    dice = np.mean([x for x in result_list['dice_1'] if x >= 0])
    try:
        auc = roc_auc_score(result_list['gt_label'], result_list['pred_label'])
    except ValueError:
        auc = -1
    acc = accuracy_score(result_list['gt_label'], result_list['pred_label'])
    precision = precision_score(result_list['gt_label'], result_list['pred_label'])
    recall = recall_score(result_list['gt_label'], result_list['pred_label'])

    result = {
        'dice': dice,
        'auc': auc,
        'acc': acc,
        'precision': precision,
        'recall': recall
    }

    if detail_info:
        return result, result_list
    else:
        return result


if __name__ == '__main__':
    dt = torch.rand(8, 1, 512, 512).float()
    gt = torch.randint(0, 2, [8, 1, 512, 512]).long()
    auc = roc_auc_score(gt.view(-1), dt.view(-1))
    gt_tensor = torch.tensor(gt).long()
    dt_tensor = torch.tensor(dt).float()
    auc1 = auc_score(gt_tensor, dt_tensor)
    print(auc)
    print(auc1)
