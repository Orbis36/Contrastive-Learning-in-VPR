from .vpr_model import build_model


def build_network(model_cfg, cache_dataset):
    model = build_model(model_cfg=model_cfg, cache_dataset=cache_dataset)
    return model
