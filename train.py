from datasets import create_dataset
from parse_config import get_parsed_args
from utils.common_utils import set_random_seed
from datasets import create_dataset
from models import build_network

import torch
import yaml

if __name__ == "__main__":

    opt, dataset_cfg, model_cfg = get_parsed_args()
    if opt.fix_random_seed:
        set_random_seed(opt.fix_random_seed)

    # TODO: Output dir and LOG dir and Wandb setting

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")
    device = torch.device("cuda" if cuda else "cpu")
    
    train_dataset = create_dataset(dataset_cfg, mode='train')
    val_dataset = create_dataset(dataset_cfg, mode='val')
    
    model = build_network(model_cfg=model_cfg)

    