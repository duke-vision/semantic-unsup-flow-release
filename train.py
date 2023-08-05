import datetime
curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

import argparse
import pprint
import os
from path import Path

from utils.logger import init_logger
from utils.config_parser import init_config

import torch
torch.backends.cudnn.benchmark = True
from utils.torch_utils import init_seed

import numpy as np

from datasets.get_dataset import get_dataset
from models.get_model import get_model
from losses.get_loss import get_loss
from trainer.get_trainer import get_trainer


def main(cfg, _log, resume=False):
    init_seed(cfg.seed)

    # prepare data
    _log.info("=> fetching img pairs.")
    train_sets, valid_sets, train_sets_epoches = get_dataset(cfg.data)
    _log.info("train sets: " + ", ".join(["{} ({} samples)".format(ds.name, len(ds)) for ds in train_sets]))
    _log.info("val sets: " + ", ".join(["{} ({} samples)".format(ds.name, len(ds)) for ds in valid_sets]))
    
    train_sets_epoches = [np.inf if e == -1 else e for e in train_sets_epoches]

    _log.info("=> setting up data loaders.")
    train_loaders, valid_loaders = [], []
    max_val_batch = 4
    
    for ds in train_sets:
        train_loader = torch.utils.data.DataLoader(
            ds, batch_size=cfg.train.batch_size,
            num_workers=cfg.train.workers, pin_memory=True, shuffle=True)
        train_loaders.append(train_loader)
    
    for ds in valid_sets:
        valid_loader = torch.utils.data.DataLoader(
            ds, batch_size=cfg.train.batch_size,
            num_workers=cfg.train.workers, pin_memory=True, shuffle=False)
        valid_loaders.append(valid_loader)
    valid_size = sum([len(l) for l in valid_loaders])

    if cfg.train.valid_size == 0:
        cfg.train.valid_size = valid_size  
    cfg.train.valid_size = min(cfg.train.valid_size, valid_size)

    # prepare model
    model = get_model(cfg.model)
    
    # prepare loss
    loss = get_loss(cfg.loss)
    
    # prepare training scipt
    trainer = get_trainer(cfg.trainer)(
        train_loaders, valid_loaders, model, loss, _log, cfg.save_root, cfg.train, resume=resume, train_sets_epoches=train_sets_epoches)
    
    trainer.train()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/kitti_base.json')
    parser.add_argument('-m', '--model', default=None)
    parser.add_argument('--exp_folder', default='other')
    parser.add_argument('-n', '--name', default=None)
    parser.add_argument('-r', '--resume', default=None)
    parser.add_argument('--n_gpu', type=int, default=2)
    parser.add_argument('--DEBUG', action='store_true')
    args = parser.parse_args()
    
    # resuming
    if args.resume is not None:
        args.resume = Path(args.resume)
        args.config = args.resume / 'config.json'

    # load config
    cfg = init_config(args.config)       
    cfg.train.n_gpu = args.n_gpu
    
    # DEBUG options
    cfg.train.DEBUG = args.DEBUG
    if args.DEBUG:
        cfg.train.update({
            'epoch_num': 5,
            'epoch_size': 4,
            'print_freq': 1,
            'record_freq': 1,
            'val_epoch_size': 1,
            'valid_size': 1
        })
        if 'stage1' in cfg.train:
            cfg.train.stage1.update({'epoch': 1})
        if 'stage2' in cfg.train:
            cfg.train.stage2.update({'epoch': 2})
        
    # load model 
    if args.model is not None:
        cfg.train.pretrained_model = args.model
        
    # init save_root: store files by curr_time
    if args.resume is not None:
        cfg.save_root = args.resume
    else:      
        if args.name is None:
            args.name = os.path.basename(args.config)[:-5]
            
        if args.DEBUG:
            cfg.save_root = Path('results/' + args.exp_folder) / '_DEBUG_' + curr_time + '_' + args.name
        else:
            cfg.save_root = Path('results/' + args.exp_folder) / curr_time + '_' + args.name

        cfg.save_root.makedirs_p()
        os.system('cp {} {}'.format(args.config, cfg.save_root / 'config.json'))
        if 'base_configs' in cfg:
            os.system('cp {} {}'.format(os.path.join(os.path.dirname(args.config), cfg.base_configs), cfg.save_root / cfg.base_configs))

    # init logger
    slurm_id = os.environ.get('SLURM_JOBID')
    slurm_path = 'results/jobs/train_{}.out'.format(slurm_id)
    if os.path.exists(slurm_path):
        os.system('ln -s {} {}'.format(slurm_path, cfg.save_root / 'slurm_{}.out'.format(slurm_id)))
    
    _log = init_logger(log_dir=cfg.save_root, filename='{}.log'.format(curr_time))
    _log.info('=> slurm jobid: {}'.format(slurm_id))
    _log.info('=> will save everything to {}'.format(cfg.save_root))
    print(cfg.save_root)

    # show configurations
    cfg_str = pprint.pformat(cfg)
    _log.info('=> configurations \n ' + cfg_str)

    main(cfg, _log, resume=args.resume is not None)
