from .dataset import DatasetTemplate
from .pittsburgh import Pittsburgh
from .msls import MSLS

__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'Pittsburgh': Pittsburgh,
    'MSLS': MSLS
}

def create_dataset(dataset_cfg, mode):
    dataset = __all__[dataset_cfg.NAME](config=dataset_cfg, mode=mode)
    return dataset