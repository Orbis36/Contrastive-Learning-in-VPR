from os.path import join
from .dataset import DatasetTemplate
from scipy.io import loadmat
from .augmentor.image_augmentor import DataAugmentor

import numpy as np

class Pittsburgh(DatasetTemplate):
    def __init__(self, config, mode):
        super().__init__(config, mode)

        assert mode in ['test', 'val', 'train'], "Mode select error"
        assert config.SPLIT in ['pitts30k', 'pitts250k'], "Pittsbugh splited error, only support 30k or 250k"
        assert config.PRETEXT_TASK in ['Regression', 'Classifiction', 'Contrastive'], "Pretext task error"
        assert config.TRAINING_TASK in ['i2i'], "Pitts dataset only support i2i task" # No seq data annotation in Pittsburgh dataset

        self.root_path = config.PATH.ROOT_PATH
        self.mergin = config.SAMPLE.MERGIN
        self.split = config.SPLIT
        self.pretext_task = config.PRETEXT_TASK
        self.dbStruct = self.parse_dbStruct(join(self.root_path, config.PATH.MAT_PATH))
        
        self.mode = mode
        self.cache = None
        self.negCache = [np.empty((0,)) for _ in range(self.dbStruct.numQ)]

        # 导入所有query-image的index，如果为测试模式则导入全部images
        self.images = [join(self.root_path, dbIm) for dbIm in self.dbStruct.dbImage]
        if self.mode == 'test':
            queries_dir = self.root_path + config.PATH.QUERIES_PATH
            self.images += [join(queries_dir, qIm) for qIm in self.dbStruct.qImage]

        augmentor = DataAugmentor(config.AUGMENTOR)
        
        if self.pretext_task == 'Contrastive':
            self.potential_negatives, self.queries, self.nontrivial_positives = super().query_preprocess(self.dbStruct)
        elif self.pretext_task == 'Classification':
            raise NotImplementedError
        elif self.pretext_task == 'Regression':
            raise NotImplementedError
        else:
            raise ValueError("Pretext task selected haven't been implemented")

    
    
    def parse_dbStruct(self, path):
        mat_path = join(path, self.split + '_' + self.check_mode + '.mat')
        mat = loadmat(mat_path)
        matStruct = mat['dbStruct'].item()
        whichSet = matStruct[0].item()

        dbImage = [f[0].item() for f in matStruct[1]]
        utmDb = matStruct[2].T

        qImage = [f[0].item() for f in matStruct[3]]
        utmQ = matStruct[4].T

        numDb = matStruct[5].item()
        numQ = matStruct[6].item()

        posDistThr = matStruct[7].item()
        posDistSqThr = matStruct[8].item()
        nonTrivPosDistSqThr = matStruct[9].item()

        return self.dbStruct(whichSet, self.split, dbImage, utmDb, qImage, 
                utmQ, numDb, numQ, posDistThr, 
                posDistSqThr, nonTrivPosDistSqThr)

    
    
    
    
