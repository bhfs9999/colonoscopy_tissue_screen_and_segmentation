import argparse

from lib.util.solver import Solver
from torch.utils.data.dataloader import DataLoader

from configs import cfg
from lib.util.transform import build_transforms
from lib.dataset import Colonoscopy, collate_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch v1.1, Colonoscopy segmentation, Testing')
    parser.add_argument(
        '--config-file',
        default="",
        metavar="FILE",
        help="path to config file",
        type=str
    )
    parser.add_argument(
        '--tta',
        help='whether use tta in test',
        default=False,
        action='store_true',
    )

    args = parser.parse_args()

    if args.config_file != '':
        cfg.merge_from_file(args.config_file)
    cfg.freeze()
    print('Using config: ', cfg)

    output_dir = cfg.OUTPUT_DIR

    transforms_test = build_transforms(cfg, is_train=False)
    dataset_test = Colonoscopy(split_file=cfg.DATA.DATASET_VALID,
                               img_root=cfg.DATA.DATA_ROOT,
                               transforms=transforms_test)
    dataloader_test = DataLoader(dataset_test, batch_size=cfg.SOLVER.BATCH_SIZE_TEST,
                                 num_workers=1, collate_fn=collate_fn)

    solver = Solver(cfg, output_dir, is_train=False)

    solver.test(dataloader_test=dataloader_test, tta=args.tta)
