from .vpr_model import build_model


def build_network(model_cfg):
    model = build_model(model_cfg=model_cfg)
    return model
