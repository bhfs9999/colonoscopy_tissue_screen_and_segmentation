import os
import numpy as np

data_root = '/data/bhfs/challenge/MICCAI2019/Colonoscopy_tissue_segment_dataset'

all_img = set()

for img_name in os.listdir(os.path.join(data_root, 'tissue-train-neg')):
    img_name = img_name.split('.')[0]
    all_img.add(img_name.strip('_mask'))

all_img = np.array(list(all_img))

train_rate = 0.8
valid_rate = 0.1
train_num = int(len(all_img) * train_rate)
valid_num = int(len(all_img) * valid_rate)
shuffled_img = np.random.permutation(all_img)

train_set = shuffled_img[:train_num]
valid_set = shuffled_img[train_num:train_num + valid_num]
test_set = shuffled_img[train_num + valid_num:]

save_root = os.path.join(data_root, 'split_neg')
if not os.path.exists(save_root):
    os.makedirs(save_root)

for set_name, data_set in zip(['train', 'valid', 'test'], [train_set, valid_set, test_set]):
    save_path = os.path.join(save_root, set_name)
    with open(save_path + '.txt', 'w') as f:
        for aline in data_set:
            f.write(aline + '\n')
