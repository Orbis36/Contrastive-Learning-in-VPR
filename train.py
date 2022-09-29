from datasets import create_dataset
from parse_config import get_parsed_args
from utils.common_utils import set_random_seed
from utils.optimization import build_scheduler, build_optimizer

from datasets import create_dataset
from models import build_network

import torch
import yaml
import os

# def train_model(model, data_loader, optimizer, total_epoch):



if __name__ == "__main__":

    opt, dataset_cfg, model_cfg = get_parsed_args()
    if opt.fix_random_seed:
        set_random_seed(opt.fix_random_seed)
    
    # path to 
    if not os.path.exists(opt.temp_weight_path):
        os.makedirs(opt.temp_weight_path)
    os.environ["TORCH_HOME"]=opt.temp_weight_path

    # TODO: Output dir and LOG dir and Wandb setting
    # TODO: Multi-Card training

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")
    device = torch.device("cuda" if cuda else "cpu")
    
    train_dataset = create_dataset(dataset_cfg, mode='train')
    val_dataset = create_dataset(dataset_cfg, mode='val')
    
    model = build_network(model_cfg=model_cfg.MODEL)
    optimizer = build_optimizer(opt_cfg=model_cfg.OPTIMIZATION)
    scheduler = build_scheduler(optimizer, model_cfg.OPTIMIZATION)
    model.train()
    # TODO：dataset细化

    for epoch in range(model_cfg.OPTIMIZATION.EPOCH):
        if opt.optim.upper() == 'SGD':
            scheduler.step(epoch)
        

    del os.environ["TORCH_HOME"]

    