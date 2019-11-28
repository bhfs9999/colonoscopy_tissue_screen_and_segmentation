from yacs.config import CfgNode as Node
import os


# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
cfg = Node()
cfg.PRO_ROOT = os.path.abspath(os.path.join(os.getcwd()))
cfg.OUTPUT_DIR = os.path.join(cfg.PRO_ROOT, 'output')


# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------
cfg.DATA = Node()
# path to split file that contain image names, with one name in one line, here use fold 0
cfg.DATA.DATASET_TRAIN = './data_split/split_4_fold/0_train.txt'
cfg.DATA.DATASET_VALID = './data_split/split_4_fold/0_valid.txt'
cfg.DATA.DATASET_TEST = './data_split/split_4_fold/0_valid.txt'  # no test data left, you can make you own test data
# path to dir that save tissue image and mask, image and mask
cfg.DATA.DATA_ROOT = '/data/bhfs/challenge/MICCAI2019/Colonoscopy_tissue_segment_dataset/tissue-train-pos-neg/'

cfg.DATA.NUM_CLS = 2  # binary segmentation here


# data transformation

# input size
cfg.DATA.SIZE_TRAIN = 512
cfg.DATA.SIZE_TEST = 512

# crop method
cfg.DATA.CROP_METHOD_TRAIN = 'random'  # random, resize
cfg.DATA.RANDOM_CROP_POS_CENTER_P = 0.  # with a probability p, that a patch is centered on a lesion area
# dense: sliding window method
cfg.DATA.CROP_METHOD_TEST = 'dense'  # random, gird, dense, resize
cfg.DATA.DENSE_CROP_STRIDE = 256
cfg.DATA.DENSE_CROP_MERGE_METHOD = 'or'  # how to merge the overlap part of patch, 'and' 'or'


cfg.DATA.RANDOM_ROTATE = True
# edge mirror padding of patch, set 0 to disable this transformation
cfg.DATA.MIRROR_PADDING = 0
# around padding of image, the image is first padded with AroundPaddingSize, and each patch is of size
# SIZE_TEST + AroundPaddingSize, but we only use SIZE_TEST in merge function, set 0 to disable this transformation
cfg.DATA.AroundPaddingSize = 0
# random shuffle rgb channel
cfg.DATA.RANDOM_SHUFFLE_CHANNEL = False
# brightness, contrast, hsv transform
cfg.DATA.COLOR_JITTER = False
cfg.DATA.BRIGHTNESS = 0.2
cfg.DATA.CONTRAST = 0.2
cfg.DATA.SATURATION = 20
cfg.DATA.HUE = 30
cfg.DATA.VALUE = 20

# down sample rate, during training and test
cfg.DATA.DOWN_SAMPLE_RATE = (1.0, )
cfg.DATA.DOWN_SAMPLE_RATE_TEST = (4.0, )

# # for ImageNet
# cfg.DATA.MEAN = (103.53, 116.28, 123.675)
# cfg.DATA.STD = (57.375, 57.12, 58.395)

# # for pos and neg sample
cfg.DATA.MEAN = (200.88868021, 173.87053633, 205.59562856)
cfg.DATA.STD = (54.10893796, 76.54162457, 44.94489184)

# for pos sample only
# cfg.DATA.MEAN = (197.67294614, 166.74599843, 202.84378699)
# cfg.DATA.STD = (55.25516756, 79.19482492, 45.39172564)


# -----------------------------------------------------------------------------
# SOLVER
# -----------------------------------------------------------------------------
cfg.SOLVER = Node()
# total patches per batch is BATCH_SIZE_TRAIN(TEST) * PATCH_NUM_PER_IMG_TRAIN(TEST) for random, grid crop
# for resize, total patches per batch is BATCH_SIZE_TRAIN/TEST
# for dense crop, BATCH_SIZE_TRAIN/TEST must be 1, patch nums vary with image size
# images per batch
cfg.SOLVER.BATCH_SIZE_TRAIN = 2
cfg.SOLVER.BATCH_SIZE_TEST = 1
# patches per image, for random, grid crop
cfg.SOLVER.PATCH_NUM_PER_IMG_TRAIN = 2
cfg.SOLVER.PATCH_NUM_PER_IMG_TEST = 2
# for sliding window, how many patches one batch during test
cfg.SOLVER.BATCH_SIZE_PER_IMG_TEST = 8

# save model, do validation, record result
cfg.SOLVER.SAVE_INTERVAL_EPOCH = 10

cfg.SOLVER.EPOCHS = 200
cfg.SOLVER.STEPS = (100, 150)  # for multi-step lr scheduler

cfg.SOLVER.LR = 0.001
cfg.SOLVER.WEIGHT_DECAY = 0.0001
cfg.SOLVER.MOMENTUM = 0.9
cfg.SOLVER.WARMUP_METHOD = 'linear'
cfg.SOLVER.WARMUP_FACTOR = 1.0 / 3
cfg.SOLVER.WARMUP_ITERS = 500
cfg.SOLVER.GAMMA = 0.1

# clip loss
cfg.SOLVER.CLIP_LOSS = False
cfg.SOLVER.CLIP_LOSS_TH = 1.


# save visualization result as image in test phase
cfg.SOLVER.DRAW = False


# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
cfg.MODEL = Node()
# model to use, see lib/modeling/factory for detail
cfg.MODEL.MODEL = 'UNet'
# whether use deep supervision, only for UNet family
cfg.MODEL.DEEP_SUP = ''
cfg.MODEL.DEEP_SUP_SE = False
cfg.MODEL.FUSION_DS = ''
cfg.MODEL.CLS_BRANCH = False

# loss to use, see lib/loss/factory for detail
cfg.MODEL.LOSS = 'DiceLoss'

# optimizer to use, see /lib/optimizer/factory for detail
cfg.MODEL.OPTIMIZER = 'SGD'

# resume iter, for resume training or test, eg. '030' for 30 epoch's checkpoint
cfg.MODEL.RESUME_ITER = ''
# loss weight for bg and fg
cfg.MODEL.LOSS_WEIGHT = (1.0, 1.0)
# for BceWithLogDiceLoss
cfg.MODEL.BCE_WEIGHT = 1.0
# gamma for focal loss
cfg.MODEL.LOSS_FL_GAMMA = 2.0
