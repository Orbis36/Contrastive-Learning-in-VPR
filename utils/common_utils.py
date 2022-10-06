import random
import numpy as np
import torch

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

def reFormatDict(batch_list):

    data_dict = defaultdict(list)
    for cur_sample in batch_list:
        for key, val in cur_sample.items():
            data_dict[key].append(val)
    return data_dict
