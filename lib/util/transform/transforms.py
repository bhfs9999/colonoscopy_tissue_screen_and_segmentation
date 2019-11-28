import cv2
import numpy as np
import random
import torch
import math

from albumentations.augmentations import functional as F
from torchvision.transforms.functional import to_tensor
from lib.util.process import valid_patch


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask=None):
        for t in self.transforms:
            if mask is None:
                img = t(img)
            else:
                img, mask = t(img, mask)

        if mask is None:
            return img
        else:
            return img, mask


class Convert2Float:
    def __call__(self, img, mask=None):
        return img.astype(np.float32), mask


class Convert2Int:
    def __call__(self, img, mask=None):
        if mask is not None:
            return img.astype(np.uint8), mask
        else:
            return img.astype(np.uint8)


class RandomSampleCrop:
    """
    Random crop patches from image with fixed size 'crop_size', given 'crop_nums', ignore white noise patch
    with probability pos_center_p, crop patch with a positive pixel at the center of the patched, otherwise, random crop
    each connective area is selected with equally probability
    """

    def __init__(self, crop_size=512, crop_nums=1, pos_center_p=0.):
        self.crop_size = crop_size
        self.crop_nums = crop_nums
        self.pos_center_p = pos_center_p

    def get_random_patch(self, img, mask):
        if mask.sum() > 0 and random.random() < self.pos_center_p:
            # crop guarantee positive pixel at the center of the patch
            _, labels = cv2.connectedComponents(mask)
            idx = random.randint(1, labels.max())
            area_yx = (labels == idx).nonzero()
            area_pixel_nums = len(area_yx[0])
            center_idx = random.randint(0, area_pixel_nums - 1)
            center_y, center_x = area_yx[0][center_idx], area_yx[1][center_idx]

            h, w = mask.shape[:2]
            y2 = int(min(h, center_y + self.crop_size / 2))
            x2 = int(min(w, center_x + self.crop_size / 2))
            y1 = int(max(0, y2 - self.crop_size))
            x1 = int(max(0, x2 - self.crop_size))
            y2 = min(y1 + self.crop_size, h)
            x2 = min(x1 + self.crop_size, w)

            img_patch = img[y1:y2, x1:x2]
            mask_patch = mask[y1:y2, x1:x2]
            return img_patch, mask_patch

        else:
            # random crop
            while True:
                h, w = img.shape[:2]
                if self.crop_size >= w:
                    x1 = 0
                else:
                    x1 = random.randint(0, w - self.crop_size)
                if self.crop_size >= h:
                    y1 = 0
                else:
                    y1 = random.randint(0, h - self.crop_size)

                x2 = min(x1 + self.crop_size, w)
                y2 = min(y1 + self.crop_size, h)
                img_patch = img[y1:y2, x1:x2]
                if valid_patch(img_patch):
                    mask_patch = mask[y1:y2, x1:x2]
                    return img_patch, mask_patch

    def __call__(self, img, mask):
        img_patches = np.zeros((self.crop_nums, self.crop_size, self.crop_size, 3)).astype(np.uint8)
        mask_patches = np.zeros((self.crop_nums, self.crop_size, self.crop_size)).astype(np.uint8)
        for i in range(self.crop_nums):
            img_patch, mask_patch = self.get_random_patch(img, mask)
            h, w = img_patch.shape[:2]
            img_patches[i, :h, :w, :], mask_patches[i, :h, :w] = img_patch, mask_patch

        return img_patches, mask_patches


class GridSampleCrop:
    def __init__(self, crop_size=512, crop_nums=16):
        """
        sample crop for validation or testing
        :param crop_size: crop size
        :param crop_num: crop nums, must bu complete square number due to gird crop
        """
        assert pow(math.sqrt(crop_nums), 2) == crop_nums, "crop num should be Complete square number"
        self.crop_size = crop_size
        self.crop_nums_per_line = math.sqrt(crop_nums)

    def get_grid(self, shape):
        """
        get center point to crop for a img
        :param shape: img shape, h *w *3
        :return: crop center point of num self.crop_nums, equal spaceing
        """
        h, w = shape[:2]

        TO_REMOVE = 1
        h_interval = int((h - self.crop_size - TO_REMOVE) / (self.crop_nums_per_line - 1))
        w_interval = int((w - self.crop_size - TO_REMOVE) / (self.crop_nums_per_line - 1))

        h_grid = range(int(self.crop_size / 2), h - int(self.crop_size / 2), h_interval)
        w_grid = range(int(self.crop_size / 2), w - int(self.crop_size / 2), w_interval)

        return [[wi, hi] for hi in h_grid for wi in w_grid]

    def do_crop(self, img, mask, x, y):
        x1 = x - int(self.crop_size / 2)
        x2 = x1 + self.crop_size
        y1 = y - int(self.crop_size / 2)
        y2 = y1 + self.crop_size
        return img[y1:y2, x1:x2], mask[y1:y2, x1:x2]

    def __call__(self, img, mask):
        crop_grid = self.get_grid(img.shape)

        patches = [self.do_crop(img, mask, *crop_xy) for crop_xy in crop_grid]
        return np.stack([b[0] for b in patches]), np.stack([b[1] for b in patches])


class DenseCrop:
    def __init__(self, crop_size=512, crop_stride=256):
        """
        crop fixed size patch densely from original image with overlap
        :param crop_size: crop size
        :param crop_stride: overlap pixel between two adjacent patch
        """
        self.crop_size = int(crop_size)
        self.crop_stride = crop_stride

    def get_grid(self, img_size):
        res = []
        cur = 0
        while True:
            x2 = min(cur + self.crop_size, img_size)
            x1 = max(0, x2 - self.crop_size)
            res.append([x1, x2])
            if x2 == img_size:
                break
            cur += self.crop_stride
        return res

    def get_xys(self, shape):
        """
        get top left and bottom right corner points to crop for a img densely
        :param shape: img shape, h *w *3
        :return: list of size crop_num * 4, top left(x1, y1) and bottom right(x2, y2) corner points to crop
        """
        h, w = shape[:2]
        h_grid = self.get_grid(h)
        w_grid = self.get_grid(w)
        xy_grid = [[wi[0], hi[0], wi[1], hi[1]] for hi in h_grid for wi in w_grid]

        return xy_grid

    @staticmethod
    def do_crop(img, mask, x1, y1, x2, y2):
        if mask is not None:
            return img[y1:y2, x1:x2], mask[y1:y2, x1:x2]
        else:
            return img[y1:y2, x1:x2]

    def __call__(self, img, mask=None):
        crop_xys = self.get_xys(img.shape)
        img_patches = np.zeros((len(crop_xys), self.crop_size, self.crop_size, 3)).astype(np.uint8)
        if mask is not None:
            mask_patches = np.zeros((len(crop_xys), self.crop_size, self.crop_size)).astype(np.uint8)
        for i, xy in enumerate(crop_xys):
            if mask is not None:
                img_patch, mask_patch = self.do_crop(img, mask, *xy)
            else:
                img_patch = self.do_crop(img, mask, *xy)
            h, w = img_patch.shape[:2]
            img_patches[i, :h, :w, :] = img_patch
            if mask is not None:
                mask_patches[i, :h, :w] = mask_patch

        if mask is not None:
            return img_patches, mask_patches
        else:
            return img_patches


class EdgePadding:
    """
    mirror padding
    """

    def __init__(self, padding=16):
        self.padding = padding

    def do_pad(self, img, mask=None):
        if self.padding == 0:
            if mask is not None:
                return img, mask
            else:
                return img
        img_t = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='reflect')
        if mask is not None:
            mask_t = np.pad(mask, ((self.padding, self.padding), (self.padding, self.padding)), mode='reflect')
            return img_t, mask_t
        else:
            return img_t

    def __call__(self, img, mask=None):
        if len(img.shape) == 3:
            return self.do_pad(img, mask)

        elif len(img.shape) == 4:
            img_t = np.zeros((img.shape[0], img.shape[1] + self.padding * 2,
                              img.shape[2] + self.padding * 2, img.shape[3])).astype(np.uint8)
            if mask is not None:
                mask_t = np.zeros((img.shape[0], img.shape[1] + self.padding * 2,
                                   img.shape[2] + self.padding * 2)).astype(np.uint8)
                for i in range(img.shape[0]):
                    img_t[i], mask_t[i] = self.do_pad(img[i], mask[i])
                return img_t, mask_t
            else:
                for i in range(img.shape[0]):
                    img_t[i] = self.do_pad(img[i])
                return img_t


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            if len(img.shape) == 3:
                if random.random() < self.p:
                    img = np.ascontiguousarray(img[::-1, :])
                    mask = np.ascontiguousarray(mask[::-1, :])
            elif len(img.shape) == 4:
                for i in range(img.shape[0]):
                    if random.random() < self.p:
                        img[i] = np.ascontiguousarray(img[i, ::-1, :])
                        mask[i] = np.ascontiguousarray(mask[i, ::-1, :])

        return img, mask


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if len(img.shape) == 3:
            if random.random() < self.p:
                img = np.ascontiguousarray(img[:, ::-1])
                mask = np.ascontiguousarray(mask[:, ::-1])
        elif len(img.shape) == 4:
            for i in range(img.shape[0]):
                if random.random() < self.p:
                    img[i] = np.ascontiguousarray(img[i, :, ::-1])
                    mask[i] = np.ascontiguousarray(mask[i, :, ::-1])

        return img, mask


class RandomRotate:
    def __call__(self, img, mask):
        if len(img.shape) == 3:
            rotate_times = random.randint(0, 3)
            img = np.ascontiguousarray(np.rot90(img, rotate_times))
            mask = np.ascontiguousarray(np.rot90(mask, rotate_times))
        if len(img.shape) == 4:
            for i in range(img.shape[0]):
                rotate_times = random.randint(0, 3)
                img[i] = np.ascontiguousarray(np.rot90(img[i], rotate_times))
                mask[i] = np.ascontiguousarray(np.rot90(mask[i], rotate_times))

        return img, mask


class RandomShuffleChannel:
    perms = ((0, 1, 2), (0, 2, 1),
             (1, 0, 2), (1, 2, 0),
             (2, 0, 1), (2, 1, 0))

    @staticmethod
    def swap_channels(img, swap):
        return img[:, :, swap]

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask=None):
        if random.random() < self.p:
            swap = self.perms[random.randint(0, len(self.perms) - 1)]
            img = self.swap_channels(img, swap)

        return img, mask


class ColorJitter:
    """Randomly change brightness, contrast, hue, saturation and value of the input image.

    Args:
        brightness ((float, float) or float): factor range for changing brightness.
            If limit is a single float, the range will be (-limit, limit). Default: 0.2.
        contrast ((float, float) or float): factor range for changing contrast.
            If limit is a single float, the range will be (-limit, limit). Default: 0.2.
        hue ((int, int) or int): range for changing hue. If hue_shift_limit is a single int, the range
            will be (-hue_shift_limit, hue_shift_limit). Default: 20.
        saturation ((int, int) or int): range for changing saturation. If sat_shift_limit is a single int,
            the range will be (-sat_shift_limit, sat_shift_limit). Default: 30.
        value ((int, int) or int): range for changing value. If val_shift_limit is a single int, the range
            will be (-val_shift_limit, val_shift_limit). Default: 20.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, brightness=0.2, contrast=0.2, hue=20, saturation=30, value=20, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.hue = hue
        self.saturation = saturation
        self.value = value
        self.p = p

    def do_color_jitter(self, img):
        alpha = 1.0 + random.uniform(-self.contrast, self.contrast)
        beta = 0.0 + random.uniform(-self.brightness, self.brightness)
        img = F.brightness_contrast_adjust(img, alpha, beta)

        hue_shift = random.uniform(-self.hue, self.hue)
        saturation_shift = random.uniform(-self.saturation, self.saturation)
        value_shift = random.uniform(-self.value, self.value)
        img = F.shift_hsv(img, hue_shift, saturation_shift, value_shift)

        return img

    def __call__(self, img, mask):
        if len(img.shape) == 3:
            if random.random() < self.p:
                img = self.do_color_jitter(img)
        elif len(img.shape) == 4:
            for i in range(img.shape[0]):
                if random.random() < self.p:
                    img[i] = self.do_color_jitter(img[i])

        return img, mask


class Rescale:
    def __init__(self, scale=255.0):
        self.scale = scale

    def __call__(self, img, mask):
        return img / self.scale, mask


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask=None):
        """
        :param img: ndarray uint8
        :param mask:  ndarray uint8
        :return: normalized img float32, ori mask
        """
        img = ((img.astype(np.float32) - self.mean) / self.std).astype(np.float32)

        if mask is not None:
            return img, mask
        else:
            return img


class Resize:
    def __init__(self, size=(512, 512)):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img, mask=None):
        img = cv2.resize(img, self.size)

        if mask is not None:
            mask = cv2.resize(mask, self.size)
            return img[None, :, :, :], mask[None, :, :]

        else:
            return img


class ToTensor:
    def __call__(self, img, mask):
        """
        convert ndarray to tensor, then rearrange axis to (B), C, H, W
        :param img: ndarray of size, (B), H, W, C
        :param mask:
        :return:
        """
        if len(img.shape) == 3:
            img_t = to_tensor(img)
            mask_t = to_tensor(mask)
        elif len(img.shape) == 4:
            img_t = torch.zeros(img.shape[0], img.shape[3], img.shape[1], img.shape[2])
            mask_t = torch.zeros(img.shape[0], img.shape[3], img.shape[1], img.shape[2])
            for i in range(img.shape[0]):
                img_t[i] = to_tensor(img[i])
                mask_t[i] = to_tensor(mask[i])

        return img_t, mask_t


class RandomDownSample:
    """
    down sample img and mask
    """

    def __init__(self, downsample_rate=(1.0,)):
        """
        random downsample image and mask, downsample scale is chosen from downsample_rate
        :param downsample_rate: tuple of int, eg. [1, 2, 3], to random downsample 1 2 or 3 scales
        """
        self.scales = [1.0 / k for k in downsample_rate]

    def __call__(self, img, mask=None):
        k = random.randint(0, len(self.scales) - 1)
        scale = self.scales[k]
        img = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if mask is not None:
            mask = cv2.resize(mask, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            return img, mask
        else:
            return img


class RearrangeChannel:
    """
    rearrange image channel from h, w, c to c, h, w
    mask from h, w to 1, h, w
    """

    def __call__(self, img, mask):
        img = img.transpose(2, 0, 1)
        if len(mask.shape) == 2:
            mask = mask[None, :, :]

        return img, mask


class ToFloat:
    def __call__(self, img, mask):
        img = img.astype(np.float32)

        return img, mask


class AroundPadding:
    def __init__(self, pad_size):
        self.pad_size = pad_size

    def __call__(self, img, mask=None):
        if self.pad_size == 0:
            if mask is not None:
                return img, mask
            else:
                return img
        img_t = np.pad(img, ((self.pad_size, self.pad_size), (self.pad_size, self.pad_size), (0, 0)), mode='reflect')
        if mask is not None:
            mask_t = np.pad(mask, ((self.pad_size, self.pad_size), (self.pad_size, self.pad_size)), mode='reflect')
            return img_t, mask_t
        else:
            return img_t
