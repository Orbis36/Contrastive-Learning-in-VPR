from .model_template import VPRModelTemplate
from math import ceil
from torch.utils.data import DataLoader, SubsetRandomSampler
from os.path import isfile, join

import torch.nn.functional as F
import numpy as np
import h5py
import torch
import faiss



class NetVLAD(VPRModelTemplate):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        self.module_list = self.build_networks(model_cfg=model_cfg)
        # self.descriptor_num = model_cfg.DESCRIPTORS_NUM
        # self.nPerImage = model_cfg.NUM_DESC_PER_IMAGE
        # self.cache_bs = model_cfg.CACHE_BS

    def forward(self, batch_dict, need_grad=True):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        return batch_dict

    def get_training_loss(self, batch_dict):
        loss_triplet = self.feature_selecter.get_feature_selected_loss(batch_dict)
        return loss_triplet