from .model_template import VPRModelTemplate

class NetVLAD(VPRModelTemplate):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        self.module_list = self.build_networks(model_cfg=model_cfg)

    def forward(self, batch_dict):

        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss = self.get_training_loss(batch_dict)
        
        return loss

    def get_training_loss(self, batch_dict):
        loss_triplet = self.feature_selecter.get_feature_selected_loss(batch_dict)
        return loss_triplet