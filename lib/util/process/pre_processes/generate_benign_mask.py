import os
import cv2
import numpy as np
from tqdm import tqdm


img_root = '../../../../data/raw_data'


if __name__ == '__main__':
    img_names = set([x.split('.')[0].strip('_mask') for x in os.listdir(img_root)])

    for name in tqdm(img_names):
        mask_path = os.path.join(img_root, '{}_mask.jpg'.format(name))
        # pos image has mask
        if os.path.exists(mask_path):
            continue
        # neg image
        else:
            image_path = os.path.join(img_root, '{}.jpg'.format(name))
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.imwrite(mask_path, mask)
