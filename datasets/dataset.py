import torch.utils.data as torch_data

class DatasetTemplate(torch_data.Dataset):
    def __init__(self):
        super().__init__()
        pass

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getitem__(self, index):
        raise NotImplementedError

    def update_subcache(self):
        pass
