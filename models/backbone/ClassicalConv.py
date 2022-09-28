import torch.nn as nn
import torchvision.models as models

class ClassicalConv(nn.Module):
    def __init__(self, model_cfg):

        encoder_name = model_cfg.NAME 

        # TODO: 这里需要让网络控制是否不需要训练，已经参数导入，为了seq模型

        self.freeze_to_layer = model_cfg.FREEZE_TO
        if encoder_name == 'vgg16':
            encoder= models.vgg16(pretrained=model_cfg.pretrained)
            layers = list(encoder.features.children())[:-2]
            if model_cfg.pretrained:
            # if using pretrained then only train conv5_1, conv5_2, and conv5_3
                for l in layers[:-5]: 
                    for p in l.parameters():
                        p.requires_grad = False

        elif encoder_name == 'alexnet':
            encoder = models.alexnet(pretrained=model_cfg.pretrained)
            layers = list(encoder.features.children())[:-2]
            if model_cfg.pretrained:
                for l in layers[:-1]:
                    for p in l.parameters():
                        p.requires_grad = False
        else:
            raise ValueError("Type used not a classical conv backbone")

        encoder = nn.Sequential(*layers)
        model = nn.Module() 
        self.model = model.add_module('encoder', encoder)
        

    def forward(self, data_dict):
        # 注意，这里看下cluster的加入的L2 norm
        all_samples = data_dict['samples']
        data_dict['samples'] = self.model(all_samples)
        return data_dict



        
        
