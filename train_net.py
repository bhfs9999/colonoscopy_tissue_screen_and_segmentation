import argparse
import os

from lib.util.solver import Solver
from torch.utils.data.dataloader import DataLoader
from lib.util.deterministic import seed_torch

from configs import cfg
from lib.util.transform import build_transforms
from lib.dataset import Colonoscopy, collate_fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch v1.1, Colonoscopy segmentation Training')
    parser.add_argument(
        '--config-file',
        default="",
        metavar="FILE",
        help="path to config file",
        type=str
    )

    args = parser.parse_args()

    if args.config_file != '':
        cfg.merge_from_file(args.config_file)
    cfg.freeze()
    print('Using config: ', cfg)

    output_dir = cfg.OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    seed_torch(0)

    transforms_train = build_transforms(cfg, is_train=True)
    dataset_train = Colonoscopy(split_file=cfg.DATA.DATASET_TRAIN,
                                img_root=cfg.DATA.DATA_ROOT,
                                transforms=transforms_train)
    dataloader_train = DataLoader(dataset_train, batch_size=cfg.SOLVER.BATCH_SIZE_TRAIN,
                                  num_workers=4, shuffle=True, collate_fn=collate_fn)

    transforms_valid = build_transforms(cfg, is_train=False)
    dataset_valid = Colonoscopy(split_file=cfg.DATA.DATASET_VALID,
                                img_root=cfg.DATA.DATA_ROOT,
                                transforms=transforms_valid,
                                )
    dataloader_valid = DataLoader(dataset_valid, batch_size=cfg.SOLVER.BATCH_SIZE_TEST,
                                  num_workers=1, collate_fn=collate_fn)

    solver = Solver(cfg, output_dir)

    solver.train(dataloader_train, dataloader_valid)
