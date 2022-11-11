from .dataset import DatasetTemplate


class OxfordFD(DatasetTemplate):
    def __init__(self, config, mode):
        super().__init__(config, mode)

    