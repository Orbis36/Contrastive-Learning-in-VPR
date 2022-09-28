from .model_template import VPRModelTemplate

class NetVLAD(VPRModelTemplate):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        print()