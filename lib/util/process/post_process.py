import numpy as np
import cv2
import torch
from torch.nn import functional as F
# import pydensecrf.densecrf as dcrf


def remove_small_region(pred_mask, th=500):
    contours_mask, _ = cv2.findContours(pred_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    left = [cont for cont in contours_mask if cv2.contourArea(cont) > th]
    canvas = np.zeros_like(pred_mask, dtype=np.uint8)
    cv2.drawContours(canvas, left, -1, 1, -1)
    return canvas


def remove_mirror_padding(pred_mask_patch, padding=16):
    """
    remove mirror padding in transformation before training
    :param pred_mask_patch: predicted mask patch, each edge has 'padding' nums mirror padding,
    tensor of shape [patch_nums, h, w]
    :param padding:
    :return:
    """
    if padding == 0:
        return pred_mask_patch

    return pred_mask_patch[:, padding: -padding, padding: -padding]


def get_grid(img_size, crop_size, crop_stride):
    res = []
    cur = 0
    while True:
        x2 = min(cur + crop_size, img_size)
        x1 = max(0, x2 - crop_size)
        res.append([x1, x2])
        if x2 == img_size:
            break
        cur += crop_stride
    return res


def merge_patch(patches, h, w, stride=256, merge_method='or', binary_th=0.5):
    """
    merge patches split from dense crop
    :param patches: patches cropped from a single image, tensor of shape [patch_nums, h, w]
    :param h: single image's height
    :param w: width
    :param stride: crop stride
    :param merge_method: how to merge overlapping part, logical or, logical and, probability mean
    :param binary_th:
    :return: merged image, [binary mask, probability map] if using prob_mean, else return [binary, binary]
    """
    patches_prob = patches.numpy()
    patches = np.where(patches > binary_th, 1, 0).astype(np.uint8)
    if merge_method == 'prob_mean':
        single_image_pred_mask = np.zeros((h, w), dtype=np.float32)
    else:
        single_image_pred_mask = np.zeros((h, w), dtype=np.uint8)

    patch_h, patch_w = patches[0].shape[:2]
    h_grid = get_grid(h, patch_h, stride)
    w_grid = get_grid(w, patch_w, stride)

    ovlap_h = patch_h - stride
    ovlap_w = patch_w - stride
    idx = 0

    for y1, y2 in h_grid:
        for x1, x2 in w_grid:
            # patch in range [0, valid_w/h] is valid, otherwise is 0, which is filled for batch inference
            valid_w = x2 - x1
            valid_h = y2 - y1
            # left top
            if x1 == 0 and y1 == 0:
                new_part_x_global, new_part_y_global = slice(x1, x2), slice(y1, y2)
                new_part_x_patch, new_part_y_patch = slice(0, valid_w), slice(0, valid_h)

            # left
            elif x1 == 0 and y1 != 0:
                new_part_x_global, new_part_y_global = slice(x1, x2), slice(y1 + ovlap_h, y2)
                new_part_x_patch, new_part_y_patch = slice(0, valid_w), slice(ovlap_h, valid_h)
            # top
            elif x1 != 0 and y1 == 0:
                new_part_x_global, new_part_y_global = slice(x1 + ovlap_w, x2), slice(y1, y2)
                new_part_x_patch, new_part_y_patch = slice(ovlap_w, valid_w), slice(0, valid_h)
            # central
            else:
                new_part_x_global, new_part_y_global = slice(x1 + ovlap_w, x2), slice(y1 + ovlap_h, y2)
                new_part_x_patch, new_part_y_patch = slice(ovlap_w, valid_w), slice(ovlap_h, valid_h)
            if merge_method == 'and':
                np.logical_and(single_image_pred_mask[y1:y2, x1:x2],
                               patches[idx, :valid_h, :valid_w],
                               out=single_image_pred_mask[y1:y2, x1:x2])
            elif merge_method == 'or':
                np.logical_or(single_image_pred_mask[y1:y2, x1:x2],
                              patches[idx, :valid_h, :valid_w],
                              out=single_image_pred_mask[y1:y2, x1:x2])
            elif merge_method == 'prob_mean':
                single_image_pred_mask[y1:y2, x1:x2] = (single_image_pred_mask[y1:y2, x1:x2] +
                                                        patches_prob[idx, :valid_h, :valid_w]) / 2
            else:
                print("merge method must be 'and', 'or', 'prob_mean', got ", merge_method)
                raise ValueError
            if merge_method == 'prob_mean':
                single_image_pred_mask[new_part_y_global, new_part_x_global] = patches_prob[idx][
                    new_part_y_patch, new_part_x_patch]
            else:
                single_image_pred_mask[new_part_y_global, new_part_x_global] = patches[idx][
                    new_part_y_patch, new_part_x_patch]
            idx += 1

    if merge_method == 'prob_mean':
        single_image_pred_mask_probs = single_image_pred_mask.copy()
        single_image_pred_mask_binary = np.where(single_image_pred_mask_probs > binary_th, 1, 0).astype(np.uint8)
        return single_image_pred_mask_binary, single_image_pred_mask_probs
    else:
        return single_image_pred_mask, single_image_pred_mask


def valid_patch(patch, threshold=160):
    """
    check whether patches is a valid patch, which is not a white noise image
    :param patch: ndarray, has the same width and height
    :param threshold: threshold under which for a bin to be a noise bin
    :return:
    """
    bin_size = int(patch.shape[0] / 16)

    for m in range(16):
        for n in range(16):
            x, y = m * bin_size, n * bin_size
            mean = patch[y: y + bin_size, x: x + bin_size].mean()
            if mean > threshold:
                return True

    return False


def fill_hole(pred_mask):
    h, w = pred_mask.shape
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    canvas[1: h+1, 1: w+1] = pred_mask.copy()

    mask = np.zeros((h + 4, w + 4), np.uint8)

    cv2.floodFill(canvas, mask, (0, 0), 1)
    canvas = canvas[1: h+1, 1: w+1].astype(np.bool)

    canvas = ~canvas | pred_mask.astype(np.uint8)

    return canvas


def resize_ndarry(img, width, height):
    """
    resize ndarray img to given width and height
    """
    img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    return img


def resize_tensor(img, width, height):
    """
    resize tensor img to given width and height
    """
    if len(img.shape) == 2:
        # convert to B, C, w, h
        img = img[None, None, :, :]
    if img.dtype == torch.uint8:
        # 0, 1 mask
        img = img.float()
        img = F.interpolate(img, size=(height, width), mode='bicubic', align_corners=False).to(torch.uint8)
    else:
        # prob mask
        img = F.interpolate(img, size=(height, width), mode='bicubic', align_corners=False)
    img = img[0, 0]
    return img


'''
def dense_crf(prob_mask, ori_image):
    """
    https://github.com/lucasb-eyer/pydensecrf#usage
    crf to refine predicted probability map. if ori_image is given, the pairwise bilateral potential
    on raw RGB values will be computed.
    for binary segmentation only
    :param prob_mask: adarray of type np.uint8, with shape [h, w]
    :param ori_image: adarray of type np.float32, with shape [h, w, 3]
    :return:
    """
    h, w = prob_mask.shape
    num_cls = 2

    probas = np.stack([1 - prob_mask, prob_mask], axis=0)  # 2,H,W
    mask = np.where(prob_mask > 0.5, 255, 0).astype(np.uint8)
    cv2.imwrite('/data/bhfs/output/123.jpg', mask)

    # output_probs = np.expand_dims(output_probs, 0)
    # output_probs = np.append(1 - output_probs, output_probs, axis=0)

    d = dcrf.DenseCRF2D(w, h, num_cls)

    U = -np.log(probas)  # h, w, np.float32
    U = U.reshape((num_cls, -1))

    U = np.ascontiguousarray(U)
    ori_image = np.ascontiguousarray(ori_image)

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=20, compat=3)
    d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=ori_image, compat=10)

    # d.addPairwiseGaussian(sxy=3, compat=3)
    # d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=img, compat=10)

    Q = d.inference(5)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

    mask = np.where(Q > 0.5, 255, 0).astype(np.uint8)
    cv2.imwrite('/data/bhfs/output/123123.jpg', mask)

    return Q
'''
