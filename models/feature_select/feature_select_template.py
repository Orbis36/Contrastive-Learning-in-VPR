
import torch.nn as nn

class FeatureSelectTemplate(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.MARGIN = model_cfg.MARGIN**0.5
        

    def build_loss(self):
        self.add_module(
            'triplet_loss_func',
            nn.TripletMarginLoss(margin=float(self.MARGIN, p=2, reduction='sum'))
        )

    def forward(self):
        raise NotImplementedError


        