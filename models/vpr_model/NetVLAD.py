from .model_template import VPRModelTemplate

class NetVLAD(VPRModelTemplate):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        print()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)