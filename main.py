""" 
main.py
Created by zenn at 2021/7/18 15:08
"""
import pytorch_lightning as pl
import argparse

import pytorch_lightning.utilities.distributed
import torch
import yaml
from easydict import EasyDict
import os

from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from datasets import get_dataset
from models import get_model
from torchstat import stat
from ptflops import get_model_complexity_info
# os.environ["NCCL_DEBUG"] = "INFO"

def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config

# def prepare_input(resolution):
#     template_points = torch.cuda.FloatTensor(1, 512, 3)
#     search_points = torch.cuda.FloatTensor(1, 1024, 3)
#     # points2cc_dist_t = torch.cuda.FloatTensor(1, 512, 9)
#     # input_dict = {
#     #     'template_points': template_points,
#     #     'search_points': search_points,
#     #     'points2cc_dist_t': points2cc_dist_t
#     # }
#     # input_dict['template_points'] = template_points
#     # return dict( template_points = template_points)
#     return dict(input_dict = [template_points, search_points])

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=50, help='input batch size')
    parser.add_argument('--epoch', type=int, default=60, help='number of epochs')
    parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
    parser.add_argument('--gpu', type=int, nargs='+', default=(0, 1), help='specify gpu devices')
    parser.add_argument('--cfg', type=str, default='./cfgs/P2B.yaml', help='the config_file')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint location')
    parser.add_argument('--log_dir', type=str, default=None, help='log location')
    parser.add_argument('--test', action='store_true', default=False, help='test mode')
    parser.add_argument('--save_top_k', type=int, default=-1,
                        help='save top k checkpoints, use -1 to checkpoint every epoch')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='check_val_every_n_epoch')
    parser.add_argument('--preloading', action='store_true', default=False, help='preload dataset into memory')

    args = parser.parse_args()
    config = load_yaml(args.cfg)
    config.update(vars(args))  # override the configuration using the value in args

    return EasyDict(config)


cfg = parse_config()

# init model
if cfg.checkpoint is None:
    net = get_model(cfg.net_model)(cfg)
    # net = net.cuda()
    # flops, params = get_model_complexity_info(net, input_res=((1, 512, 3), (1, 1024, 3), (1, 512, 9)),
    #                                           input_constructor=prepare_input,
    #                                           as_strings=True, print_per_layer_stat=False)
    # print('      - Flops:  ' + flops)
    # print('      - Params: ' + params)


else:
    net = get_model(cfg.net_model).load_from_checkpoint(cfg.checkpoint, config=cfg)
if not cfg.test:
    # dataset and dataloader
    train_data = get_dataset(cfg, type='train', split='train')
    # val_data = get_dataset(cfg, type='test', split='test')
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, num_workers=cfg.workers, shuffle=True,
                              pin_memory=True)
    # val_loader = DataLoader(val_data, batch_size=1, num_workers=cfg.workers, collate_fn=lambda x: x, pin_memory=True)
    checkpoint_callback = ModelCheckpoint(monitor='success/test', mode='max', save_last=True,
                                          save_top_k=cfg.save_top_k)

    # init trainer
    trainer = pl.Trainer(gpus=cfg.gpu, accelerator='ddp', max_epochs=cfg.epoch, resume_from_checkpoint=cfg.checkpoint,
                         callbacks=[checkpoint_callback], default_root_dir=cfg.log_dir,
                         check_val_every_n_epoch=cfg.check_val_every_n_epoch)
    # trainer.fit(net, train_loader, val_loader)
    trainer.fit(net, train_loader)
else:
    test_data = get_dataset(cfg, type='test', split='test')
    test_loader = DataLoader(test_data, batch_size=1, num_workers=cfg.workers, collate_fn=lambda x: x, pin_memory=True)

    trainer = pl.Trainer(gpus=cfg.gpu, accelerator='ddp')
    trainer.validate(net, test_loader)
