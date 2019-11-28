import os
from collections import OrderedDict
import time

from tqdm import tqdm
from scipy import ndimage as ndi
import numpy as np
import cv2
import torch

from lib.modeling import get_model
from lib.util.transform import build_transforms
from lib.util.process import tta as test_time_aug
from lib.util.process.post_process import remove_small_region, resize_ndarry, merge_patch
from configs import cfg


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def load_model(model, resume_path):
    checkpoint = torch.load(resume_path)
    model_dcit = checkpoint['model']
    model_dcit = strip_prefix_if_present(model_dcit, prefix='module.')
    model.load_state_dict(model_dcit)
    return model


def get_kfold_models(cfg, best_dice=True):
    """
    get k folds model from one config file, each fold get one model, with best dice when best_dice=True,
    else get best auc point
    :param cfg:
    :param fold_nums:
    :param best_dice:
    :return:
    """
    models = []
    weight_root = cfg.OUTPUT_DIR

    for fold_name in os.listdir(weight_root):
        model = get_model(cfg, pre_train=False)
        file_name = 'best_valid_dice.pkl' if best_dice else 'best_valid_auc.pkl'
        resume_path = os.path.join(weight_root, fold_name, file_name)
        print(resume_path)
        model = load_model(model, resume_path)
        if torch.cuda.is_available():
            model.cuda()
        else:
            print('CUDA not support!!')
            raise ValueError
        model.eval()
        models.append(model)

    return models


def get_transformed_img(img_path, transform):
    img_path = os.path.join(img_path)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_info = {'h': img.shape[0],
                'w': img.shape[1]}

    img_t = transform(img)
    img_t = torch.from_numpy(img_t).permute(0, 3, 1, 2).float()

    return img_t, img_info


def dump_pred_mask(mask, dump_path):
    if len(mask.shape) == 3:
        mask = mask.squeeze(0)
    mask = (mask * 255).astype(np.uint8)
    cv2.imwrite(dump_path, mask)


def get_cls_score(pred_mask):
    pred_area = pred_mask.sum()
    total_area = pred_mask.shape[-1] * pred_mask.shape[-2]
    score = float(pred_area) / total_area
    return score


if __name__ == '__main__':
    cfg_files = [
        './configs/submit/unet16_fold4_down4_layer4.yaml',
        './configs/submit/layer4_dicelossv1_dsv2_down4.yaml',
        './configs/submit/unet19_fold4_down4_layer4.yaml',
        './configs/submit/unet19_fold4_down4_layer4_ds.yaml',
    ]
    batch_size = 1
    down_sample_rate = 4.0
    cfg.DATA.SIZE_TEST = 2048
    cfg.DATA.DENSE_CROP_STRIDE = 2048
    cfg.DATA.DENSE_CROP_MERGE_METHOD = 'or'
    binary_th = 0.2
    binary_th_cls = 0.96
    remove_area = 500
    remove_area_cls = 500

    models = []

    tic = time.time()
    print('Loading model')
    for cfg_file in cfg_files:
        # get best dice checkpoint model for k fold
        cfg_current_model = cfg.clone()
        cfg_current_model.merge_from_file(cfg_file)
        models += get_kfold_models(cfg_current_model, best_dice=True)
    print('load model: ', time.time() - tic)

    inference_transform = build_transforms(cfg_current_model, is_train=False)
    input_root = '/input'
    mask_output_root = '/output/predictions'
    cls_output_path = '/output/predict.csv'
    if not os.path.exists(mask_output_root):
        os.makedirs(mask_output_root)

    pred_cls_labels = {}

    # inference
    with torch.no_grad():
        for img_name in tqdm(os.listdir(input_root), desc='Inference ...'):
            img_path = os.path.join(input_root, img_name)
            img_patches, img_info = get_transformed_img(img_path, inference_transform)
            patch_nums, _, h, w = img_patches.shape
            pred_mask_patches = torch.zeros(patch_nums, h, w).float()

            for i in range(0, patch_nums, batch_size):
                start_idx = i
                end_index = min(i + batch_size, patch_nums)
                inputs = img_patches[start_idx: end_index]

                all_model_outputs = []
                tta_inputs = {}

                for tta_name, tta_f in test_time_aug.ttas.items():
                    # tta
                    tta_inputs[tta_name] = tta_f(inputs)
                for model in models:
                    # each model
                    tta_outputs = []
                    for tta_name, tta_input in tta_inputs.items():
                        # each tta
                        tta_input = tta_input.cuda()
                        outputs = model(tta_input)
                        outputs = outputs.cpu()
                        detta_f = test_time_aug.dettas[tta_name]
                        tta_output = detta_f(outputs)
                        tta_outputs.append(tta_output)

                    one_model_outputs = torch.cat(tta_outputs, 1).mean(1)  # B, H, W
                    all_model_outputs.append(one_model_outputs.unsqueeze(1))

                final_outputs = torch.cat(all_model_outputs, 1).mean(1)  # B. H. W
                pred_mask_patches[start_idx: end_index] = final_outputs

            ori_h, ori_w = img_info['h'], img_info['w']
            scale_h = round(ori_h / down_sample_rate)
            scale_w = round(ori_w / down_sample_rate)

            whole_pred_mask_binary, _ = \
                merge_patch(pred_mask_patches,
                            h=scale_h, w=scale_w,
                            stride=cfg.DATA.DENSE_CROP_STRIDE,
                            merge_method=cfg.DATA.DENSE_CROP_MERGE_METHOD,
                            binary_th=binary_th)

            whole_pred_mask_binary_cls, _ = \
                merge_patch(pred_mask_patches,
                            h=scale_h, w=scale_w,
                            stride=cfg.DATA.DENSE_CROP_STRIDE,
                            merge_method=cfg.DATA.DENSE_CROP_MERGE_METHOD,
                            binary_th=binary_th_cls)

            # for dice
            # need to restore to original size when using downsample
            whole_pred_mask_binary = resize_ndarry(whole_pred_mask_binary, ori_w, ori_h)  # H, W

            # post process
            whole_pred_mask_binary = ndi.binary_fill_holes(whole_pred_mask_binary).astype(np.uint8)
            whole_pred_mask_binary = remove_small_region(whole_pred_mask_binary, remove_area)

            # save pred mask
            mask_save_path = os.path.join(mask_output_root, img_name)
            dump_pred_mask(whole_pred_mask_binary, mask_save_path)

            # for cls
            # need to restore to original size when using downsample
            whole_pred_mask_binary_cls = resize_ndarry(whole_pred_mask_binary_cls, ori_w, ori_h)  # H, W

            # post process
            whole_pred_mask_binary_cls = ndi.binary_fill_holes(whole_pred_mask_binary_cls).astype(np.uint8)
            whole_pred_mask_binary_cls = remove_small_region(whole_pred_mask_binary_cls, remove_area_cls)

            # record cls label
            pred_cls_label = get_cls_score(whole_pred_mask_binary_cls)
            pred_cls_labels[img_name] = pred_cls_label

    with open(cls_output_path, 'w') as f:
        f.write('image_name,score\n')
        for name, label in pred_cls_labels.items():
            f.write('{},{}\n'.format(name, label))
