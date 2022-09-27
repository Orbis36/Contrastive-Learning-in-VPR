import cv2
import torch
import numpy as np

from functools import partial

class DataAugmentor(object):
    def __init__(self, cur_cfg) -> None:
        # TODO: augment 队列，多线程操作
        self.data_augmentor_queue = []
        cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
        self.data_augmentor_queue.append(cur_augmentor)
    
    def image_normalize(self, data_dict=None, config=None):
        if data_dict is None:
            assert len(config['STD']) == len(config['MEAN']) == 3, "image norm should only have 3 channels"
            self.std = config['STD']
            self.mean = config['MEAN']
            return partial(self.random_world_rotation, config=config)
        img = data_dict['image'] / 255.0
        img -= np.array(self.mean)
        data_dict['image'] = img / np.array(self.std)

        return img

    def forward(self, data_dict):
        for _aug in self.data_augmentor_queue:
            data_dict = _aug(data_dict)
        return data_dict

        



    
