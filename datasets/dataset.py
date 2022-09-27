import torch.utils.data as torch_data
from collections import namedtuple

class DatasetTemplate(torch_data.Dataset):
    def __init__(self):
        super().__init__()
        self.dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset', 
        'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
        'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getitem__(self, index):
        raise NotImplementedError

    def update_subcache(self):
        pass
