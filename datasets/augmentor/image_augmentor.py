import cv2
import torch
import numpy as np
import faiss
import torch.nn as nn

from functools import partial


class DataAugmentor(object):
    def __init__(self, cur_cfg) -> None:
        # TODO: augment 队列，多线程操作
        self.data_augmentor_queue = []
        aug_config_list = cur_cfg if isinstance(cur_cfg, list) else cur_cfg.AUG_CONFIG_LIST
        for cur_cfg in aug_config_list:
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)
    
    def image_normalize(self, data_dict=None, config=None):
        if data_dict is None:
            assert len(config.STD) == len(config.MEAN) == 3, "image norm should only have 3 channels"
            self.std = np.asarray(config.STD)
            self.mean = np.asarray(config.MEAN)
            return partial(self.image_normalize, config=config)
        img = data_dict['image'] / 255.0
        img -= np.array(self.mean)
        data_dict['image'] = img / np.array(self.std)
        return data_dict

    def forward(self, data_dict):
        for _aug in self.data_augmentor_queue:
            data_dict = _aug(data_dict)
        return data_dict

        



    
