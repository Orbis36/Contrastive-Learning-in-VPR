from turtle import back
import torch
import os
import torch.nn as nn

from models import backbone
from models import feature_select
from models import post_matching
from models import model_head

class VPRModelTemplate(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.module_topology = ['backbone', 'feature_selecter', 'post_matching', 'model_head']
        # self.cache_save_path = os.path.join(os.getcwd(), 'weight_data/cache_pth')
        # os.makedirs(self.cache_save_path) if not os.path.exists(self.cache_save_path) else print("cache_pth exist")

    def build_networks(self, model_cfg):
        model_cfg['module_list'] = []
        for module_name in self.module_topology:
            module, model_cfg = getattr(self, 'build_%s' % module_name)(
                model_cfg=model_cfg
            )
            self.add_module(module_name, module)
        return model_cfg['module_list']
    
    def build_backbone(self, model_cfg):
        backbone_module = backbone.__all__[model_cfg.BACKBONE.NAME](model_cfg.BACKBONE)
        out_dim = backbone_module.out_dim
        model_cfg['FEATURE_SELECT']['BACKBONE_OUT_DIM'] = out_dim
        model_cfg['module_list'].append(backbone_module)
        return backbone_module, model_cfg

    def build_feature_selecter(self, model_cfg):
        selector = feature_select.__all__[model_cfg.FEATURE_SELECT.NAME](model_cfg.FEATURE_SELECT)
        model_cfg['module_list'].append(selector)
        return selector, model_cfg

    def build_post_matching(self, model_cfg):
        # 允许模型可以不做精细匹配
        # TODO: 看看这里如何表示 找不到key
        if model_cfg.get('feature_matcher', None) is None:
            return None, model_cfg
        post_matcher = post_matching.__all__[model_cfg.feature_matcher.NAME](model_cfg.feature_matcher)
        model_cfg['module_list'].append(post_matcher)
        return post_matcher, model_cfg

    def build_model_head(self, model_cfg):
        # 允许模型没有head，这里为了方便多模态与class-based单模态放在这里
        if model_cfg.get('head', None) is None:
            return None, model_cfg
        model_head = model_head.__all__[model_cfg.model_head.NAME](model_cfg.model_head)
        model_cfg['module_list'].append(model_head)
        return model_head, model_cfg
