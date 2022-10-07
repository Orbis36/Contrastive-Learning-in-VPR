import random
import numpy as np
import torch
import os

from collections import defaultdict

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_to_gpu(batch_dict):

     for key, val in batch_dict.items():
        if key in ['samples', 'image']:
            batch_dict[key] = torch.from_numpy(val).float().cuda()
        elif key in ['nQuery', 'nNeg', 'negCounts', 'p_n_label', 'idx']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        elif key in ['bs', 'nNegUse']:
            continue
        else:
            print('Error in collate_batch: key=%s' % key)
            raise TypeError

def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu

def reFormatDict(batch_list):

    data_dict = defaultdict(list)
    for cur_sample in batch_list:
        for key, val in cur_sample.items():
            data_dict[key].append(val)
    return data_dict

def saveCheckPoint(model=None, optimizer=None, epoch=None, recalls=None, filename='checkpoint'):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None
    state = {'epoch': epoch, 'recalls': recalls, 'model_state': model_state, 'optimizer_state': optim_state}
    filename = os.path.join(model.weight_save_path, '{}_{}.pth'.format(epoch, filename))
    torch.save(state, filename)
