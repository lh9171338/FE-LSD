import os
from yacs.config import CfgNode
import argparse


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--gpu', type=int, help='gpu id')
    parser.add_argument('-a', '--arch', type=str, default='FE-HAWP', choices=['FE-HAWP', 'FE-ULSD'], help='model arch')
    parser.add_argument('-p', '--pretrained_model_name', type=str, help='pretrained model name')
    parser.add_argument('-l', '--last_epoch', type=int, help='last epoch')
    parser.add_argument('-b', '--train_batch_size', type=int, help='training batch size')
    parser.add_argument('-d', '--dataset_name', type=str, help='dataset name')
    parser.add_argument('-t', '--score_thresh', type=float, help='score threshold')
    parser.add_argument('-s', '--save_image', action='store_true', help='save image')
    parser.add_argument('-w', '--with_clear', action='store_true', help='plot with clear image')
    parser.add_argument('-e', '--evaluate', action='store_true', help='evaluate')
    parser.add_argument('-m', '--model_name', type=str, help='model name')
    parser.add_argument('-c', '--config_file', type=str, help='config filename')
    parser.add_argument('--config_path', type=str, default='config', help='config path')

    args = parser.parse_args()
    args_dict = vars(args)
    args_list = []
    for key, value in zip(args_dict.keys(), args_dict.values()):
        if value is not None:
            args_list.append(key)
            args_list.append(value)

    if args.config_file is None:
        args.config_file = f'{args.arch}.yaml'
    yaml_file = os.path.join(args.config_path, args.config_file)
    cfg = CfgNode.load_cfg(open(yaml_file))
    cfg.merge_from_list(args_list)

    if cfg.model_name == '':
        cfg.model_name = f'{cfg.arch}_{cfg.dataset_name}.pkl'
    cfg.log_path = f'{cfg.log_path}/{os.path.splitext(cfg.model_name)[0]}'
    cfg.dataset_path = os.path.join(cfg.dataset_path, cfg.dataset_name)
    cfg.output_path = os.path.join(cfg.output_path, cfg.dataset_name + f'-{os.path.splitext(cfg.model_name)[0]}')
    cfg.figure_path = os.path.join(cfg.figure_path, cfg.dataset_name)

    cfg.image_size = tuple(cfg.image_size)
    cfg.heatmap_size = tuple(cfg.heatmap_size)

    cfg.freeze()

    # Print cfg
    for k, v in cfg.items():
        print(f'{k}: {v}')

    return cfg

