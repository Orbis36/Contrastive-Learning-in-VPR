import torch
import torch.nn as nn

class VPRModelTemplate(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.module_topology = ['backbone', 'feature_select, post_matching']

    def build_networks(self, model_cfg):
        # model_cfg 可以被看作上游下游模块通信工具
        for module_name in self.module_topology:
            module, model_cfg = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_cfg
            )
            self.add_module(module_name, module)
    
    def build_backbone(self, model_info_dict):
        

        