import datetime
import os
import time
import random
# from pympler import tracker, summary, muppy
import gc

from tqdm import tqdm

from lib.loss import get_loss
from lib.modeling import get_model
from lib.util.checkpoint import CheckPointer
from lib.util.evaluation import do_evaluation
from lib.util.process.post_process import resize_ndarry, merge_patch, remove_mirror_padding, remove_small_region
from lib.optimizer import get_optimizer
from lib.util.deterministic import get_random_state, set_random_state
from lib.util.process import tta as test_time_aug

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import make_grid
import numpy as np
import cv2
from scipy import ndimage as ndi


class Solver:
    def __init__(self, cfg, output_dir, is_train=True):
        self.cfg = cfg
        self.output_dir = output_dir

        self.is_train = is_train
        if is_train:
            self.loss = get_loss(cfg)

        self.model = get_model(cfg).float()
        self._initialize_model_device()

    def _initialize_model_device(self):
        """
        get gpu or cpu devices, then copy model to the devices
        :return:
        """
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            if torch.cuda.device_count() > 1:
                print('Using', torch.cuda.device_count(), 'GPUs')
                self.model = nn.DataParallel(self.model)
            else:
                print('Using Single GPU')
        else:
            self.device = torch.device('cpu')
            print('cuda is not available, using cpu')

        self.model.to(self.device)

    def vis_img_grid(self, img, gt_mask, dt_mask, normalized=True, tb_format=True):
        """
        visualization image , gt mask and predict mask as a 2*2 grid,
        left top: ori image,
        right top: ori image with gt mask overlay,
        left bottom: ori image with pred mask overlay,
        right bottom: ori image with pred mask and gt mask overlay
        to save to tensorboard
        img is transformed, so we need to multiply std and add mean
        to transform to tensorboard style(float in range 0-1,  or uint8 in range 0-255)
        :param img: ndarray with size 3 * h * w
        :param gt_mask: ndarray with size h * w
        :param dt_mask: ndarray with h * w
        :param normalized: whether img is normalized, need to multiply std, add mean
        :param tb_format: whether return tensorboard style image, which is a tensor of shape [3, H, W], otherwise
        return cv2.imwrite format image, of shape [H, W, 3]
        """

        if len(gt_mask.shape) == 3:
            gt_mask = gt_mask.squeeze(0)
        if len(dt_mask.shape) == 3:
            dt_mask = dt_mask.squeeze(0)

        img = img.transpose(1, 2, 0).astype(np.float32)  # h, w, 3
        if normalized:
            # restore origin pixel
            img = img * self.cfg.DATA.STD + self.cfg.DATA.MEAN
            img = img.clip(0, 255).astype(np.uint8)

        dt_mask[dt_mask > 0.5] = 1
        dt_mask[dt_mask <= 0.5] = 0
        dt_mask = dt_mask.astype(np.uint8)
        vis_img = np.tile(img, (4, 1, 1, 1)).astype(np.uint8)  # 4, h, w, c

        # draw gt mask in blue
        vis_img[1, :, :, 2] = np.where(gt_mask == 1, 255, img[..., 2])
        vis_img[3, :, :, 2] = np.where(gt_mask == 1, 255, img[..., 2])

        # draw pred mask in red
        vis_img[2, :, :, 0] = np.where(dt_mask == 1, 255, img[..., 0])
        vis_img[3, :, :, 0] = np.where(dt_mask == 1, 255, img[..., 0])

        vis_img = torch.from_numpy(vis_img).permute(0, 3, 1, 2)  # 4, 3, h, w
        vis_img = make_grid(vis_img, nrow=2)

        if not tb_format:
            vis_img = vis_img.permute(1, 2, 0).numpy()
            vis_img = vis_img[:, :, (2, 1, 0)]

        return vis_img

    def vis_img(self, img, gt_mask, dt_mask, normalized=True):
        """
        visualization image, overlap gt and pred mask on image, return result
        :param img: ndarray with size 3 * h * w
        :param gt_mask: ndarray with size h * w
        :param dt_mask: ndarray with h * w
        :param normalized:  whether img is normalized, need to multiply std, add mean
        :return:
        """
        if len(gt_mask.shape) == 3:
            gt_mask = gt_mask.squeeze(0)
        if len(dt_mask.shape) == 3:
            dt_mask = dt_mask.squeeze(0)

        img = img.transpose(1, 2, 0).astype(np.float32)  # h, w, 3
        if normalized:
            # restore origin pixel
            img = img * self.cfg.DATA.STD + self.cfg.DATA.MEAN
            img = img.clip(0, 255).astype(np.uint8)

        dt_mask[dt_mask <= 0.5] = 0
        dt_mask[dt_mask > 0.5] = 1
        dt_mask.astype(np.uint8)
        gt_mask = gt_mask.astype(np.uint8)

        # draw gt mask in blue and pred mask in green
        img[..., 2] = np.where(gt_mask == 1, 255, img[..., 2])
        img[..., 1] = np.where(dt_mask == 1, 255, img[..., 1])

        # transform rgb to bgr for saving
        img = img[..., (2, 1, 0)]

        return img

    def get_tb_dir(self):
        """
        get tensorboard log dir in output_dir, reuse if exist
        :return:
        """
        files = os.listdir(self.output_dir)
        if not files:
            return None
        for file in files:
            if os.path.isdir(os.path.join(self.output_dir, file)):
                print('Resume from exiting tensorboard log dir: ', os.path.join(self.output_dir, file))
                return os.path.join(self.output_dir, file)
        return None

    def train(self, dataloader_train, dataloader_valid=None, dataloader_test=None, dataloader_rd_crop_valid=None):
        """
        do train
        print train log and save visualization result on training set per epoch
        do validation and visualization per cfg.SOLVER.SAVE_INTERVAL_EPOCH if dataloader_valid is passed
        :param dataloader_train:
        :param dataloader_valid:
        :param dataloader_test: to see whether valid and test is synchronize
        :param dataloader_rd_crop_valid: dataloader has the same setting with train dataloader but using valid set,
        used in order to compare performance on train set and validation set
        :return:
        """
        optimizer = get_optimizer(name=self.cfg.MODEL.OPTIMIZER,
                                  params=self.model.parameters(),
                                  lr=self.cfg.SOLVER.LR,
                                  momentum=self.cfg.SOLVER.MOMENTUM,
                                  weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, verbose=True, patience=10)

        checkpointer = CheckPointer(self.model, optimizer, scheduler, self.output_dir)
        extra_checkpoint_data = checkpointer.load(self.cfg.MODEL.RESUME_ITER)

        if 'random_state' in extra_checkpoint_data.keys():
            print('Setting random state from checkpoint')
            set_random_state(extra_checkpoint_data['random_state'])

        max_epochs = self.cfg.SOLVER.EPOCHS
        start_epoch = extra_checkpoint_data['epoch'] if 'epoch' in extra_checkpoint_data.keys() else 0

        tb_log_dir = self.get_tb_dir()
        if tb_log_dir is None:
            tb_log_dir = os.path.join(self.output_dir, datetime.datetime.now().strftime('%b%d_%H-%M-%S'))
            print('Creating new tensorboard log dir: ', tb_log_dir)
        tb_writer = SummaryWriter(log_dir=tb_log_dir, purge_step=start_epoch)

        start_time = time.time()
        save_arguments = {}
        pos_nums = 0
        neg_nums = 0
        best_valid_dice = extra_checkpoint_data['best_valid_dice'] \
            if 'best_valid_dice' in extra_checkpoint_data.keys() else 0
        best_valid_auc = extra_checkpoint_data['best_valid_auc'] \
            if 'best_valid_auc' in extra_checkpoint_data.keys() else 0

        for epoch in range(start_epoch, max_epochs):
            # train one epoch
            return_dict = self.train_one_epoch(epoch, dataloader_train, optimizer)

            pos_nums += return_dict['pos_num']
            neg_nums += return_dict['neg_num']

            # one epoch finished
            # 1. record epoch
            # need +1, because this epoch has finished
            save_arguments['epoch'] = epoch + 1

            # 2. calculate one epoch loss, update lr if not go down in n epochs
            loss_one_epoch = np.mean(return_dict['train_losses'])
            scheduler.step(loss_one_epoch)

            # 3. log training information, write tensorboard
            interval = time.time() - start_time
            start_time = time.time()

            eta_seconds = interval * (max_epochs - start_epoch)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            current_lr = optimizer.param_groups[0]['lr']
            print('{}, Epoch: {:<3d} loss: {:.4f}  interval: {:<8} eta: ({}) lr: {:.6f} pos_num_rate: {:.4f}'.
                  format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                         epoch,
                         loss_one_epoch,
                         '{:.3f}s'.format(interval),
                         eta_string,
                         current_lr,
                         float(pos_nums) / float(pos_nums + neg_nums)))

            tb_writer.add_scalar('loss/train', loss_one_epoch, global_step=epoch)
            tb_writer.add_scalar('lr', current_lr, global_step=epoch)

            # 4. vis on train, may cause two large tensorboard file
            vis_image = self.vis_img_grid(img=return_dict['vis_image'], gt_mask=return_dict['vis_gt_mask'],
                                          dt_mask=return_dict['vis_dt_mask'])
            tb_writer.add_image(tag='rd_crop/train', img_tensor=vis_image, global_step=epoch)

            # 5. eval on train and valid set
            with torch.no_grad():
                train_metrics = do_evaluation(return_dict['eval_gt_mask'], return_dict['eval_dt_mask'])
                for name, value in train_metrics.items():
                    tb_writer.add_scalar('{}/train'.format(name), value, global_step=epoch)
                # print('Training result: ', train_metrics)

                # inference on valid set and eval, using dataloader_rd_crop_valid
                if dataloader_rd_crop_valid is not None:
                    rd_valid_loss = self.inference_rd_crop_valid(dataloader_rd_crop_valid, epoch)
                    rd_valid_loss = float(np.mean(rd_valid_loss))

                    tb_writer.add_scalar('loss/rd_valid', rd_valid_loss, global_step=epoch)
                    print('Valid random crop loss: ', rd_valid_loss)

            # every SAVE_INTERVAL_EPOCH
            # for debug
            # if epoch % self.cfg.SOLVER.SAVE_INTERVAL_EPOCH == 0:
            if epoch != 0 and epoch % self.cfg.SOLVER.SAVE_INTERVAL_EPOCH == 0:
                # 1. if have dataloader_valid, do validation, vis and record metric results in tensorboard
                # vis may cause two large tensorboard file
                if dataloader_valid is not None:
                    gt_masks, pred_masks, vis_images, _, valid_loss = self.inference(dataloader_valid,
                                                                                     epoch, post_process=True,
                                                                                     record_loss=True)

                    vis_idx = random.randint(0, len(vis_images) - 1)
                    vis_image = self.vis_img_grid(vis_images[vis_idx], gt_masks[vis_idx], pred_masks[vis_idx],
                                                  normalized=False)
                    tb_writer.add_image(tag='whole/valid', img_tensor=vis_image, global_step=epoch)

                    valid_metrics = do_evaluation(gt_masks, pred_masks)
                    for name, value in valid_metrics.items():
                        tb_writer.add_scalar('metrics_whole_{}/valid'.format(name), value, global_step=epoch)
                    tb_writer.add_scalar('loss/valid', valid_loss, global_step=epoch)
                    print('Valid result: ', valid_metrics)
                    print('Valid loss: ', valid_loss)

                    # 2. save best valid dice point
                    valid_dice_1 = valid_metrics['dice']
                    if valid_dice_1 > best_valid_dice:
                        best_valid_dice = valid_dice_1
                        print('Saving best valid dice point')
                        extra_checkpoint_data['best_valid_dice'] = best_valid_dice
                        extra_checkpoint_data['best_valid_auc'] = best_valid_auc
                        extra_checkpoint_data['random_state'] = get_random_state()
                        checkpointer.save('best_valid_dice', **save_arguments)
                        # save best auc point
                    valid_auc = valid_metrics['auc']
                    if valid_auc > best_valid_auc:
                        best_valid_auc = valid_auc
                        print('Saving best valid auc point')
                        extra_checkpoint_data['best_valid_dice'] = best_valid_dice
                        extra_checkpoint_data['best_valid_auc'] = best_valid_auc
                        extra_checkpoint_data['random_state'] = get_random_state()
                        checkpointer.save('best_valid_auc', **save_arguments)

                    del gt_masks, pred_masks, vis_images, valid_loss

                if dataloader_test is not None:
                    # 3. eval on test dataset
                    # vis may cause two large tensorboard file
                    gt_masks, pred_masks, vis_images, _, test_loss = self.inference(dataloader_test,
                                                                                    epoch, post_process=True,
                                                                                    record_loss=True)

                    vis_idx = random.randint(0, len(vis_images) - 1)
                    vis_image = self.vis_img_grid(vis_images[vis_idx], gt_masks[vis_idx], pred_masks[vis_idx],
                                                  normalized=False)
                    tb_writer.add_image(tag='whole/test', img_tensor=vis_image, global_step=epoch)


                    test_metrics = do_evaluation(gt_masks, pred_masks)
                    for name, value in test_metrics.items():
                        tb_writer.add_scalar('metrics_whole_{}/test'.format(name), value, global_step=epoch)
                    tb_writer.add_scalar('loss/test', test_loss, global_step=epoch)
                    print('Test result: ', test_metrics)
                    print('Test loss: ', test_loss)

                # 4. save model
                extra_checkpoint_data['random_state'] = get_random_state()
                checkpointer.save("model_epoch_{:03d}".format(epoch), **save_arguments)

    def train_one_epoch(self, epoch, dataloader_train, optimizer):
        """
        do one epoch train procedure
        :param epoch:
        :param dataloader_train:
        :param optimizer:
        :return:
        """
        self.model.train()
        pos_num = 0
        neg_num = 0
        train_losses = []
        return_dict = {}

        for image, mask, _, in tqdm(dataloader_train, desc='Training epoch {:4d}'.format(epoch), ncols=0):
            # image: B,3,H,W; mask: B,H,W
            if len(mask.shape) == 3:
                mask = mask.unsqueeze(1)
            pos_num += int((mask == 1).sum())
            neg_num += int((mask == 0).sum())
            image = image.to(self.device)
            mask = mask.to(self.device)
            logits = self.model(image)

            loss = self.loss(mask, logits)
            train_losses.append(float(loss))

            # back propagation
            loss.backward()

            # clip loss out of range
            if self.cfg.SOLVER.CLIP_LOSS:
                clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.SOLVER.CLIP_LOSS_TH, norm_type=2)

            optimizer.step()
            optimizer.zero_grad()

            if 'eval_gt_mask' not in return_dict.keys():
                with torch.no_grad():
                    self.model.eval()
                    dt_masks = self.model(image)
                    return_dict['eval_gt_mask'] = [np.array(x.cpu()) for x in mask]
                    return_dict['eval_dt_mask'] = [np.array(x.detach().cpu()) for x in dt_masks]
                    return_dict['vis_image'] = np.array(image[0].cpu())
                    return_dict['vis_gt_mask'] = np.array(mask[0].cpu())
                    return_dict['vis_dt_mask'] = np.array(dt_masks[0].detach().cpu())
                    self.model.train()

        return_dict['train_losses'] = train_losses
        return_dict['pos_num'] = pos_num
        return_dict['neg_num'] = neg_num

        self.model.eval()

        return return_dict

    def inference_rd_crop_valid(self, dataloader, epoch):
        """
        inference on inference_rd_crop_valid dataset, only inference the sample patch nums with training data loader
        :param dataloader:
        :param epoch:
        :return:
        """
        self.model.eval()

        valid_loss = []

        for image, mask, _ in tqdm(dataloader, desc='Random valid epoch {:4d}'.format(epoch), ncols=0):
            # image: B,3,H,W; mask: B,H,W
            if len(mask.shape) == 3:
                mask = mask.unsqueeze(1)

            image = image.to(self.device)
            mask = mask.to(self.device)
            logits = self.model(image)  # B, 1, H, W

            loss = self.loss(mask, logits)
            valid_loss.append(float(loss))

        self.model.train()

        return valid_loss

    def inference(self, dataloader, epoch=None, post_process=False, record_loss=False, is_tta=False):
        """
        inference on valid or test dataset
        use sliding window if cfg.DATA.CROP_METHOD_TEST is set to 'dense'
        :param dataloader:
        :param epoch:
        :param post_process: whether do post process, eg. fill hole
        :param record_loss: whether to calculate loss
        :param is_tta: whether use test time augmentation
        :return: gt_masks, pred_masks, vis_images, list of ndarray
        """
        self.model.eval()

        pred_masks_binary = []
        pred_masks_probas = []
        gt_masks = []
        vis_images = []
        image_names = []
        inference_loss = []

        epoch = 'test' if epoch is None else epoch

        with torch.no_grad():
            for image, gt_mask, index in tqdm(dataloader, desc='Epoch {}, inference....'.format(epoch)):
                # image: B, 3, H, W
                # mask: B, H, W
                # sliding window
                if self.cfg.DATA.CROP_METHOD_TEST == 'dense':
                    # batch size for dense crop is 1
                    gt_mask_ori = dataloader.dataset.get_mask(index[0])
                    image_name = dataloader.dataset.get_name(index[0])
                    image_names.append(image_name)
                    gt_masks.append(gt_mask_ori)
                    img_ori = dataloader.dataset.get_img(index[0]).transpose(2, 0, 1)
                    vis_images.append(img_ori)
                    # cut each image into patches, forward, merge result, evaluate
                    test_batch_size = self.cfg.SOLVER.BATCH_SIZE_PER_IMG_TEST
                    patch_nums, _, h, w = image.shape
                    pred_patches_mask = torch.zeros(patch_nums, h, w).float()

                    for i in range(0, patch_nums, test_batch_size):
                        start_idx = i
                        end_idx = min(i + test_batch_size, patch_nums)
                        test_patch_batch = image[start_idx: end_idx]
                        test_gt_mask_patch = gt_mask[start_idx: end_idx]
                        test_patch_batch = test_patch_batch.to(self.device)
                        if is_tta:
                            test_patch_batch_mask = []
                            for tta_name, tta_f in test_time_aug.ttas.items():
                                input_t = tta_f(test_patch_batch)
                                dt_mask_t = self.model(input_t)  # B, 1, H, W
                                detta_f = test_time_aug.dettas[tta_name]
                                test_patch_batch_mask.append(detta_f(dt_mask_t))
                            test_patch_batch_mask = torch.cat(test_patch_batch_mask, 1).mean(1)  # B, H, W
                        else:
                            test_patch_batch_mask = self.model(test_patch_batch).squeeze(1)  # B, H, W
                        pred_patches_mask[start_idx: end_idx] = test_patch_batch_mask

                        if record_loss:
                            if len(test_gt_mask_patch.shape) == 3:
                                test_gt_mask_patch = test_gt_mask_patch.unsqueeze(1)
                            if len(test_patch_batch_mask.shape) == 3:
                                test_patch_batch_mask = test_patch_batch_mask.unsqueeze(1)
                            loss = self.loss(test_gt_mask_patch.cuda(), test_patch_batch_mask)
                            inference_loss.append(float(loss))

                    ori_h, ori_w = gt_mask_ori.shape[:2]
                    scale_h = round(ori_h / self.cfg.DATA.DOWN_SAMPLE_RATE_TEST[0])
                    scale_w = round(ori_w / self.cfg.DATA.DOWN_SAMPLE_RATE_TEST[0])

                    pred_patches_mask = remove_mirror_padding(pred_patches_mask, padding=self.cfg.DATA.MIRROR_PADDING)

                    single_image_pred_mask_binary, single_image_pred_mask_probas = \
                        merge_patch(pred_patches_mask,
                                    h=scale_h, w=scale_w,
                                    stride=self.cfg.DATA.DENSE_CROP_STRIDE,
                                    merge_method=self.cfg.DATA.DENSE_CROP_MERGE_METHOD)

                    # need to restore to original size when using downsample
                    single_image_pred_mask_binary = resize_ndarry(single_image_pred_mask_binary, ori_w, ori_h)  # H, W
                    single_image_pred_mask_probas = resize_ndarry(
                        single_image_pred_mask_probas, ori_w, ori_h).clip(min=1e-7)  # H, W
                    # crf not good
                    # single_image_pred_mask_probas = dense_crf(single_image_pred_mask_probas,
                    #                                           img_ori.transpose(1, 2, 0).astype(np.uint8))
                    if post_process:
                        single_image_pred_mask_binary = ndi.binary_fill_holes(
                            single_image_pred_mask_binary).astype(np.uint8)
                        single_image_pred_mask_binary = remove_small_region(single_image_pred_mask_binary)
                    pred_masks_binary.append(single_image_pred_mask_binary[None, :, :])  # 1, H, W
                    pred_masks_probas.append(single_image_pred_mask_probas[None, :, :])  # 1, H, W

                elif self.cfg.DATA.CROP_METHOD_TEST == 'resize':
                    hws = []
                    for img_index in index:
                        gt_mask_ori = dataloader.dataset.get_mask(img_index)
                        gt_masks.append(gt_mask_ori)
                        hws.append(gt_mask_ori.shape)
                        img_ori = dataloader.dataset.get_img(img_index).transpose(2, 0, 1)
                        vis_images.append(img_ori)
                        image_name = dataloader.dataset.get_name(img_index)
                        image_names.append(image_name)

                    image = image.to(self.device)
                    pred_mask_batch = self.model(image)  # B, 1, H, W

                    if record_loss:
                        if len(gt_mask.shape) == 3:
                            gt_mask = gt_mask.unsqueeze(1)
                        if len(pred_mask_batch.shape) == 3:
                            pred_mask_batch = pred_mask_batch.unsqueeze(1)
                        loss = self.loss(gt_mask.to(self.device), pred_mask_batch)
                        inference_loss.append(float(loss))

                    for i, single_dt in enumerate(pred_mask_batch):
                        single_dt = single_dt.data.cpu().numpy()[0]
                        single_dt = resize_ndarry(single_dt, hws[i][1], hws[i][0])
                        single_dt[single_dt > 0.5] = 1
                        single_dt[single_dt <= 0.5] = 0
                        single_dt = single_dt.astype(np.uint8)
                        if post_process:
                            single_dt = ndi.binary_fill_holes(single_dt).astype(np.uint8)
                            single_dt = remove_small_region(single_dt)
                        pred_masks_binary.append(single_dt[None, :, :])
                else:
                    print('use dense crop or resize in test and validation')
                    raise ValueError

        self.model.train()

        if record_loss:
            inference_loss = np.mean(inference_loss)

        return gt_masks, pred_masks_binary, vis_images, image_names, inference_loss

    def test(self, dataloader_test, tta):
        """
        test on test data set
        :param dataloader_test:
        :param tta:
        :return:
        """
        checkpointer = CheckPointer(self.model, save_dir=self.output_dir)
        checkpointer.load(resume_iter=self.cfg.MODEL.RESUME_ITER)

        self.model.eval()
        gt_masks, pred_masks, vis_images, image_names, _ = self.inference(dataloader_test, post_process=True,
                                                                          record_loss=False, is_tta=tta)

        with torch.no_grad():
            metric_result, result_per_image = do_evaluation(gt_masks, pred_masks, detail_info=True)

        print('Test result:', metric_result)

        dice_1_per_image = result_per_image['dice_1']
        gt_label_per_image = result_per_image['gt_label']
        pred_label_per_image = result_per_image['pred_label']

        if self.cfg.SOLVER.DRAW:
            print('Saving visualization image...')
            vis_root = os.path.join(self.output_dir, 'vis')
            if not os.path.exists(vis_root):
                os.makedirs(vis_root)
            dice_text_dict = dict()
            gt_label_text_dict = dict()
            pred_label_text_dict = dict()
            tta_tag = '_tta' if tta else ''
            for i, image_name in enumerate(tqdm(image_names)):
                pred_mask = pred_masks[i]
                dice_1 = dice_1_per_image[i]
                gt_label = gt_label_per_image[i]
                pred_label = pred_label_per_image[i]
                result_img = self.vis_img(vis_images[i], gt_masks[i], pred_mask, normalized=False)
                save_path_img = os.path.join(vis_root, 'dice_{:.3f}_gt{}_pred{}_{}{}.jpg'.format(dice_1,
                                                                                                 gt_label,
                                                                                                 pred_label,
                                                                                                 image_name,
                                                                                                 tta_tag))
                cv2.imwrite(save_path_img, result_img)
                save_path_mask = os.path.join(vis_root, 'dice_{:.3f}_gt{}_pred{}_{}_mask{}.jpg'.format(dice_1,
                                                                                                       gt_label,
                                                                                                       pred_label,
                                                                                                       image_name,
                                                                                                       tta_tag))
                pred_mask = pred_mask.squeeze(0)
                pred_mask[pred_mask > 0.5] = 1
                pred_mask[pred_mask <= 0.5] = 0
                mask_to_save = (pred_mask * 255).astype(np.uint8)
                cv2.imwrite(save_path_mask, mask_to_save)
                dice_text_dict[image_name] = dice_1
                gt_label_text_dict[image_name] = gt_label
                pred_label_text_dict[image_name] = pred_label

            with open(os.path.join(vis_root, 'result{}.csv'.format(tta_tag)), 'w') as f:
                f.write('name, dice, auc\n')
                for name, dice in dice_text_dict.items():
                    f.write('{}, {}, {}, {}\n'.format(name, dice, gt_label_text_dict[name], pred_label_text_dict[name]))

