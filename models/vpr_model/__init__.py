from .model_template import VPRModelTemplate
from .NetVLAD import NetVLAD

__all__ = {
    'VPRModelTemplate': VPRModelTemplate,
    'NetVLAD': NetVLAD
}

def build_model(model_cfg):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg
    )

    return model