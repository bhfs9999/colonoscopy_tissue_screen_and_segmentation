import os

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

data_root = '/data/bhfs/challenge/MICCAI2019/Colonoscopy_tissue_segment_dataset'
dirs = ['tissue-train-pos', 'tissue-train-neg']

for dir in dirs:
    data_path = os.path.join(data_root, dir)
    for img_name in os.listdir(data_path):
        img_path = os.path.join(data_path, img_name)
        img = Image.open(img_path)