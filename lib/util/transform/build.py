from .transforms import *


def build_transforms(cfg, is_train=True):
    """
    build transforms for training, validation and test set, according to config file
    """
    transforms = [
        Convert2Int()
    ]

    if is_train:
        transforms.append(RandomDownSample(downsample_rate=cfg.DATA.DOWN_SAMPLE_RATE))
        crop_size = int(cfg.DATA.SIZE_TRAIN - cfg.DATA.MIRROR_PADDING * 2)
        # train tf, crop, color aug, shuffle ch, rotate, flip, normalize
        # crop patch of fix size from large image
        if cfg.DATA.CROP_METHOD_TRAIN == 'random':
            transforms.append(RandomSampleCrop(crop_size=crop_size,
                                               crop_nums=cfg.SOLVER.PATCH_NUM_PER_IMG_TRAIN,
                                               pos_center_p=cfg.DATA.RANDOM_CROP_POS_CENTER_P))
        # or simply resize image
        elif cfg.DATA.CROP_METHOD_TRAIN == 'resize':
            transforms.append(Resize(size=crop_size))
        else:
            print('DATA.CROP_METHOD_TRAIN must be random or resize')
            raise ValueError

        transforms.append(EdgePadding(padding=cfg.DATA.MIRROR_PADDING))

        if cfg.DATA.COLOR_JITTER:
            transforms.append(ColorJitter(brightness=cfg.DATA.BRIGHTNESS,
                                          contrast=cfg.DATA.CONTRAST,
                                          saturation=cfg.DATA.SATURATION,
                                          hue=cfg.DATA.HUE,
                                          value=cfg.DATA.VALUE,
                                          p=0.5))
        if cfg.DATA.RANDOM_SHUFFLE_CHANNEL:
            transforms.append(RandomShuffleChannel(p=0.5))

        if cfg.DATA.RANDOM_ROTATE:
            transforms.append(RandomRotate())

        transforms += [
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
        ]

    # transforms for validation or test
    else:
        transforms.append(RandomDownSample(downsample_rate=cfg.DATA.DOWN_SAMPLE_RATE_TEST))
        crop_size = int(cfg.DATA.SIZE_TEST - cfg.DATA.MIRROR_PADDING * 2 + cfg.DATA.AroundPaddingSize * 2)
        transforms.append(AroundPadding(pad_size=cfg.DATA.AroundPaddingSize))
        # valid or test tf, crop then normalize
        # grid crop
        if cfg.DATA.CROP_METHOD_TEST == 'grid':
            transforms.append(GridSampleCrop(crop_size=crop_size,
                                             crop_nums=cfg.SOLVER.PATCH_NUM_PER_IMG_TEST))
        # dense crop
        elif cfg.DATA.CROP_METHOD_TEST == 'dense':
            assert cfg.SOLVER.BATCH_SIZE_TEST == 1, 'test batch size must be 1 when using dense crop'
            transforms.append(DenseCrop(crop_size=crop_size,
                                        crop_stride=cfg.DATA.DENSE_CROP_STRIDE))
        elif cfg.DATA.CROP_METHOD_TEST == 'random':
            transforms.append(RandomSampleCrop(crop_size=crop_size,
                                               crop_nums=cfg.SOLVER.PATCH_NUM_PER_IMG_TEST))
        # simply resize image
        elif cfg.DATA.CROP_METHOD_TEST == 'resize':
            transforms.append(Resize(size=crop_size))
        else:
            print('DATA.CROP_METHOD_TEST must be grid, dense or resize')
            raise ValueError

        transforms.append(EdgePadding(padding=cfg.DATA.MIRROR_PADDING))

    transforms += [
        # Rescale(scale=255.0),
        Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)]
    # transforms.append(ToTensor())
    transforms = Compose(transforms)

    return transforms


def build_rd_crop_valid_transform(cfg):
    """
    build random crop for validation, this is used to compare performance on train and valid set under the same setting
    :param cfg:
    :return:
    """
    crop_size = int(cfg.DATA.SIZE_TRAIN - cfg.DATA.MIRROR_PADDING * 2)
    transforms = [
        Convert2Int(),
        RandomDownSample(downsample_rate=cfg.DATA.DOWN_SAMPLE_RATE),
        RandomSampleCrop(crop_size=crop_size,
                         crop_nums=cfg.SOLVER.PATCH_NUM_PER_IMG_TRAIN,
                         pos_center_p=cfg.DATA.RANDOM_CROP_POS_CENTER_P),
        EdgePadding(padding=cfg.DATA.MIRROR_PADDING),
        Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
    ]

    # transforms.append(ToTensor())
    transforms = Compose(transforms)

    return transforms
