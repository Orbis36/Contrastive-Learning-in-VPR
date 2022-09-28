from .vpr_model import build_detector


def build_network(model_cfg):
    model = build_detector(model_cfg=model_cfg)
    return model
