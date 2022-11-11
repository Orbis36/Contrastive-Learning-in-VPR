import imp
from re import I
from .dataset import DatasetTemplate
from .pittsburgh import Pittsburgh
from .msls import MSLS
from .oxfordFD import OxfordFD
from .nordland import NordLand

__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'Pittsburgh': Pittsburgh,
    'MSLS': MSLS,
    'Oxford-FD': OxfordFD,
    'NordLand': NordLand
}

def create_dataset(dataset_cfg, mode):
    dataset = __all__[dataset_cfg.NAME](config=dataset_cfg, mode=mode)
    return dataset