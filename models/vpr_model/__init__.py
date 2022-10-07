from .model_template import VPRModelTemplate
from .NetVLAD import NetVLAD

__all__ = {
    'VPRModelTemplate': VPRModelTemplate,
    'NetVLAD': NetVLAD
}

def build_model(model_cfg, cache_dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg,
        cache_dataset=cache_dataset
    )

    return model