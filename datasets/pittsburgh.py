from os.path import join
from .dataset import DatasetTemplate
from scipy.io import loadmat
from .augmentor.image_augmentor import DataAugmentor

import torch.utils.data as data
import torch
import numpy as np
import math
import random
import cv2

class Pittsburgh(DatasetTemplate):
    def __init__(self, config, mode):
        super().__init__(config, mode)

        assert mode in ['test', 'val', 'train'], "Mode select error"
        assert config.SPLIT in ['pitts30k', 'pitts250k'], "Pittsbugh splited error, only support 30k or 250k"
        assert config.PRETEXT_TASK in ['Regression', 'Classifiction', 'Contrastive'], "Pretext task error"
        assert config.TRAINING_TASK in ['i2i'], "Pitts dataset only support i2i task" # No seq data annotation in Pittsburgh dataset

        self.device = config.DEVICE
        self.root_path = config.PATH.ROOT_PATH
        self.margin = config.SAMPLE.MERGIN
        self.split = config.SPLIT
        self.nNeg = config.TRAINING.MOST_N_NEG
        self.pretext_task = config.PRETEXT_TASK
        self.bs = config.TRAINING.BATCH_SIZE
        self.workers = config.TRAINING.WORKERS
        
        # 在pittbugrh数据集中各个weight相等
        self.dbStruct = self.parse_dbStruct(join(self.root_path, config.PATH.MAT_PATH))

        # 分别导入db和q的images
        self.dbImages = np.asarray([join(self.root_path, dbIm) for dbIm in self.dbStruct.dbImage], dtype=object)
        queries_dir = self.root_path + config.PATH.QUERIES_PATH
        self.qImages = np.asarray([join(queries_dir, qIm) for qIm in self.dbStruct.qImage], dtype=object)

        # 图像数据增强
        self.augmentor = DataAugmentor(config.AUGMENTOR)
        
        if self.pretext_task == 'Contrastive':
            self.nonNegIdx, self.qIdx, self.pIdx = super().query_preprocess(self.dbStruct)
        elif self.pretext_task == 'Classification':
            raise NotImplementedError
        elif self.pretext_task == 'Regression':
            raise NotImplementedError
        else:
            raise ValueError("Pretext task selected haven't been implemented")
        
        self.weights = np.ones(len(self.qIdx))

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

    def new_epoch(self):
        # find how many subset we need to do 1 epoch
        self.nCacheSubset = math.ceil(len(self.qIdx) / self.cached_queries)

        # get all indices
        arr = np.arange(len(self.qIdx))

        # apply positive sampling of indices
        arr = random.choices(arr, self.weights, k=len(arr))

        # calculate the subcache indices
        self.subcache_indices = np.array_split(arr, self.nCacheSubset)
        
        # reset subset counter
        self.current_subset = 0
    
    def collate_fn(self, batch):
        """Creates mini-batch tensors from the list of tuples (query, positive, negatives).

        Args:
            batch: list of tuple (query, positive, negatives).
                - query: torch tensor of shape (3, h, w).
                - positive: torch tensor of shape (3, h, w).
                - negative: torch tensor of shape (n, 3, h, w).
        Returns:
            query: torch tensor of shape (batch_size, 3, h, w).
            positive: torch tensor of shape (batch_size, 3, h, w).
            negatives: torch tensor of shape (batch_size, n, 3, h, w).
        """
        batch_dict = {}
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            return None, None, None, None, None

        query, positive, negatives, indices = zip(*batch)

        query = data.dataloader.default_collate(query)
        positive = data.dataloader.default_collate(positive)
        negatives = torch.cat(negatives, 0)
        # 统计各个bs的大小
        negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
        data_input = torch.cat([query, positive, negatives])

        batch_dict['nQuery'] = query.shape[0]
        batch_dict['nNeg'] = negatives.shape[0]
        batch_dict['negCounts'] = negCounts
        batch_dict['samples'] = data_input
        return batch_dict

    def __getitem__(self, idx):
        # 这里需要用batch dict的形式过augmentor
        # get triplet
        triplet, target = self.triplets[idx]

        # get query, positive and negative idx
        qidx = triplet[0]
        pidx = triplet[1]
        nidx = triplet[2:]

        # load images into triplet list
        negImage = np.stack([cv2.imread(self.dbImages[idx]) for idx in nidx],axis=0)
        pImage = cv2.imread(self.dbImages[pidx])[np.newaxis, ...]
        qImage = cv2.imread(self.qImages[qidx])[np.newaxis, ...]

        print()


        return query, positive, negatives, [qidx, pidx] + nidx

    