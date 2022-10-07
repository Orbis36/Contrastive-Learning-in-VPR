from datasets import create_dataset
from parse_config import get_parsed_args
from utils.common_utils import set_random_seed, load_to_gpu, saveCheckPoint
from utils.optimization import build_scheduler, build_optimizer

from datasets import create_dataset
from models import build_network
from torch.utils.data import DataLoader
from tqdm.auto import trange, tqdm
from test import test_model

import torch
import yaml
import os


if __name__ == "__main__":

    opt, dataset_cfg, model_cfg = get_parsed_args()
    
    if opt.fix_random_seed:
        set_random_seed(opt.fix_random_seed)
    
    if not os.path.exists(opt.temp_weight_path):
        os.makedirs(opt.temp_weight_path)
    os.environ["TORCH_HOME"]=opt.temp_weight_path

    # TODO: Output dir and LOG dir and Wandb setting
    # TODO: Multi-Card training
    # TODO: Combine opt config and model_cfg

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")
    device = torch.device("cuda" if cuda else "cpu")
    dataset_cfg.DEVICE = device
    
    train_dataset = create_dataset(dataset_cfg, mode='train')
    val_dataset = create_dataset(dataset_cfg, mode='val')
    
    model = build_network(model_cfg=model_cfg.MODEL, cache_dataset=train_dataset)
    model.cuda()
    # 初始化，是导入新权重并全部冻结以做seq2seq模型

    optimizer = build_optimizer(model, opt_cfg=model_cfg.OPTIMIZATION)
    scheduler = build_scheduler(optimizer, model_cfg.OPTIMIZATION)

    not_improved = 0
    best_score = 0
    for epoch in range(model_cfg.OPTIMIZATION.EPOCH):
        # TODO：这里按照不同pretext task应该写不同，分类时应该在此处划分新子集
        train_dataset.new_epoch()
        for subIter in trange(train_dataset.nCacheSubset, desc='Cache refresh'.rjust(15), position=1):
            # A single epoch
            tqdm.write('====> Building Cache')
            train_dataset.update_subcache(model)
            training_data_loader = DataLoader(dataset=train_dataset, num_workers=opt.threads,
                                            batch_size=train_dataset.bs, shuffle=True,
                                            collate_fn=train_dataset.collate_fn, pin_memory=cuda)
            model.train()
            for iteration, batch_dict in enumerate(training_data_loader):
                load_to_gpu(batch_dict)
                batch_dict = model(batch_dict)
                loss = model.get_training_loss(batch_dict)
                loss.backward()
                optimizer.step()
        if opt.optim.upper() == 'SGD':
            scheduler.step(epoch)

        if (epoch % int(dataset_cfg.TRAINING.TEST_EVERY)) == 0:
            recalls = test_model(model, val_dataset, opt.threads, cuda=device=='gpu', device=device)
            is_best = recalls[5] > best_score
            if is_best:
                not_improved = 0
                best_score = recalls[5]
                saveCheckPoint(model=model, optimizer=optimizer, epoch=epoch, recalls=recalls)
            else:
                not_improved += 1
            if opt.patience > 0 and not_improved > (opt.patience / int(dataset_cfg.TRAINING.TEST_EVERY)):
                break
    
    del os.environ["TORCH_HOME"]
    torch.cuda.empty_cache()

    