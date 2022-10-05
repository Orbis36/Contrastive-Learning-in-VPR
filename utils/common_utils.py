import random
import numpy as np
import torch


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
        elif key in ['nQuery', 'nNeg', 'negCounts']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            print('Error in collate_batch: key=%s' % key)
            raise TypeError