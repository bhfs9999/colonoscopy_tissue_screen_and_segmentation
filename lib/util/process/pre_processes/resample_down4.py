import numpy as np
import cv2
import os
import random
from collections import defaultdict
from tqdm import tqdm


def get_patch_from_centerxy(cx, cy, img_h, img_w, size):
    if size > img_h:
        y1 = 0
        y2 = img_h
    else:
        y1 = int(max(0, cy - size / 2))
        y2 = int(min(img_h, y1 + size))
        y1 = int(max(0, y2 - size))
    if size > img_w:
        x1 = 0
        x2 = img_w
    else:
        x1 = int(max(0, cx - size / 2))
        x2 = int(min(img_w, x1 + size))
        x1 = int(max(0, x2 - size))

    return x1, y1, x2, y2


def cut_patch(img, mask, x1, y1, x2, y2, patch_size):
    patch_img = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    patch_mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
    valid_h = y2 - y1
    valid_w = x2 - x1
    patch_img[:valid_h, : valid_w, :] = img[y1: y2, x1: x2, :]
    patch_mask[:valid_h, : valid_w] = mask[y1: y2, x1: x2] * 255

    return patch_img, patch_mask


pos_rate = 0.6
center_rate = 0.3
valid_neg_tissue_rate = 0.02
crop_size = 512
min_dist = 100
scale = 1 / 4

root_path = '/data/bhfs/challenge/MICCAI2019/Colonoscopy_tissue_segment_dataset/tissue-train-pos-neg'
patch_output_dir = '/data/bhfs/challenge/MICCAI2019/Colonoscopy_tissue_segment_dataset/patch_down4'
vis_output_dir = '/data/bhfs/challenge/MICCAI2019/Colonoscopy_tissue_segment_dataset/patch_down4_vis'

if not os.path.exists(vis_output_dir):
    os.makedirs(vis_output_dir)
if not os.path.exists(patch_output_dir):
    os.makedirs(patch_output_dir)

total_pos_center = 0
total_pos_remain = 0
total_neg = 0
for name in tqdm(os.listdir(root_path)):
    center_pos_count = 0
    remain_pos_count = 0
    neg_count = 0
    if 'mask' in name or 'otsu' in name:
        continue
    name = name.strip('.jpg')
    img_name = name + '.jpg'
    mask_name = name + '_mask.jpg'
    otsu_name = name + '_otsu.jpg'

    img = cv2.imread(os.path.join(root_path, img_name), cv2.IMREAD_COLOR)
    mask = cv2.imread(os.path.join(root_path, mask_name), cv2.IMREAD_GRAYSCALE)
    mask = np.where(mask > 127, 1, 0).astype(np.uint8)
    otsu = cv2.imread(os.path.join(root_path, otsu_name), cv2.IMREAD_GRAYSCALE)
    otsu = np.where(otsu > 127, 1, 0).astype(np.uint8)

    img = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    otsu = cv2.resize(otsu, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    h, w = mask.shape

    img_vis_save = img.copy()
    img_vis_save[:, :, 1] = np.where(mask > 0, 255, img_vis_save[:, :, 1])

    pos_area = (mask > 0).sum()
    valid_area = valid_neg_tissue_rate * crop_size * crop_size

    # 正样本
    if pos_area > 0:
        tissue_area = (otsu > 0).sum()
        pos_area = (mask > 0).sum()
        neg_tissue_area = tissue_area - pos_area
        pos_crop_num = int(pos_area / (min_dist * min_dist))
        neg_crop_num = max(int(neg_tissue_area / (min_dist * min_dist) / 10), 1)

        pos_center_crop_num = int(pos_crop_num * center_rate)
        pos_remain_num = pos_crop_num - pos_center_crop_num

        # 采样center crop的patch
        _, labels = cv2.connectedComponents(mask)
        # each area's sample weight is its pixel count
        weights = [(labels == label).sum() for label in range(1, labels.max() + 1)]
        try:
            sample_area_idx_list = random.choices(range(1, labels.max() + 1), weights=weights, k=pos_center_crop_num)
        except IndexError:
            print(name)
        sample_area_times = defaultdict(int)

        # 确保每个区域都被采样一次
        for i in range(1, labels.max() + 1):
            sample_area_times[i] = 1
        for idx in sample_area_idx_list:
            sample_area_times[idx] += 1

        for idx, times in sample_area_times.items():
            area_yx = (labels == idx).nonzero()
            area_pixel_nums = len(area_yx[0])
            for _ in range(times):
                center_pos_count += 1
                center_idx = random.randint(0, area_pixel_nums - 1)
                center_y, center_x = area_yx[0][center_idx], area_yx[1][center_idx]

                x1, y1, x2, y2 = get_patch_from_centerxy(center_x, center_y, h, w, crop_size)
                cv2.rectangle(img_vis_save, (x1, y1), (x2, y2), (0, 0, 255), 3)

                patch_img, patch_mask = cut_patch(img, mask, x1, y1, x2, y2, crop_size)
                patch_name = img_name.strip('.jpg') + '_{}_{}_{}_{}_pos_pos_center'.format(x1, y1, x2, y2)
                patch_img_name = patch_name + '.jpg'
                patch_mask_name = patch_name + '_mask.jpg'
                cv2.imwrite(os.path.join(patch_output_dir, patch_img_name), patch_img)
                cv2.imwrite(os.path.join(patch_output_dir, patch_mask_name), patch_mask)

        # 采样非center的阳性patch
        while pos_remain_num > 0:
            center_y = random.randint(0, h - 1)
            center_x = random.randint(0, w - 1)
            x1, y1, x2, y2 = get_patch_from_centerxy(center_x, center_y, h, w, crop_size)

            if (mask[y1: y2, x1: x2]).sum() > 10:
                cv2.rectangle(img_vis_save, (x1, y1), (x2, y2), (0, 0, 255), 3)
                pos_remain_num -= 1
                remain_pos_count += 1

                patch_img, patch_mask = cut_patch(img, mask, x1, y1, x2, y2, crop_size)
                patch_name = img_name.strip('.jpg') + '_{}_{}_{}_{}_pos_pos'.format(x1, y1, x2, y2)
                patch_img_name = patch_name + '.jpg'
                patch_mask_name = patch_name + '_mask.jpg'
                cv2.imwrite(os.path.join(patch_output_dir, patch_img_name), patch_img)
                cv2.imwrite(os.path.join(patch_output_dir, patch_mask_name), patch_mask)

        # 采样完全阴性样本
        # 防止没有完全阴性样本
        try_limit = 100 * neg_crop_num
        while neg_crop_num > 0 and try_limit > 0:
            try_limit -= 1
            center_y = random.randint(0, h - 1)
            center_x = random.randint(0, w - 1)
            x1, y1, x2, y2 = get_patch_from_centerxy(center_x, center_y, h, w, crop_size)

            # 没有阳性区域且不是全白的背景区域
            if (mask[y1: y2, x1: x2]).sum() == 0 and (otsu[y1: y2, x1: x2]).sum() > valid_area:
                cv2.rectangle(img_vis_save, (x1, y1), (x2, y2), (0, 0, 255), 3)
                neg_crop_num -= 1
                neg_count += 1

                patch_img, patch_mask = cut_patch(img, mask, x1, y1, x2, y2, crop_size)
                patch_name = img_name.strip('.jpg') + '_{}_{}_{}_{}_pos_neg'.format(x1, y1, x2, y2)
                patch_img_name = patch_name + '.jpg'
                patch_mask_name = patch_name + '_mask.jpg'
                cv2.imwrite(os.path.join(patch_output_dir, patch_img_name), patch_img)
                cv2.imwrite(os.path.join(patch_output_dir, patch_mask_name), patch_mask)

    # 负样本
    else:
        tissue_area = (otsu > 0).sum()
        neg_crop_num = max(int(tissue_area / (min_dist * min_dist) / 5), 1)

        while neg_crop_num > 0:
            center_y = random.randint(0, h - 1)
            center_x = random.randint(0, w - 1)
            x1, y1, x2, y2 = get_patch_from_centerxy(center_x, center_y, h, w, crop_size)

            # 不是全白的背景区域
            if (otsu[y1: y2, x1: x2]).sum() > valid_area:
                cv2.rectangle(img_vis_save, (x1, y1), (x2, y2), (0, 0, 255), 3)
                neg_crop_num -= 1
                neg_count += 1

                patch_img, patch_mask = cut_patch(img, mask, x1, y1, x2, y2, crop_size)
                patch_name = img_name.strip('.jpg') + '_{}_{}_{}_{}_neg'.format(x1, y1, x2, y2)
                patch_img_name = patch_name + '.jpg'
                patch_mask_name = patch_name + '_mask.jpg'
                cv2.imwrite(os.path.join(patch_output_dir, patch_img_name), patch_img)
                cv2.imwrite(os.path.join(patch_output_dir, patch_mask_name), patch_mask)

    cv2.imwrite(os.path.join(vis_output_dir, img_name), img_vis_save)
    print('pos center: {}, pos remain: {}, neg: {}, pos ratio: {}, dense: {}'.format(
        center_pos_count, remain_pos_count, neg_count,
        (center_pos_count + remain_pos_count) / (center_pos_count + remain_pos_count + neg_count),
        h * w / (remain_pos_count + neg_count + center_pos_count)))

    total_pos_center += center_pos_count
    total_pos_remain += remain_pos_count
    total_neg += neg_count

print('total:', total_neg + total_pos_remain + total_pos_center)
print(total_pos_center, total_pos_remain, total_neg)
