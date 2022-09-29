
import torch.nn as nn

class FeatureSelectTemplate(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.criterion = nn.TripletMarginLoss(margin=model_cfg.MARGIN**0.5, p=2, reduction='sum')

    def get_feature_selected_loss(batch_dict):
        pass

    def forward(self):
        raise NotImplementedError


        