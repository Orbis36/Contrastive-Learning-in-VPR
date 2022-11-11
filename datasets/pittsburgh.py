from os.path import join
from .dataset import DatasetTemplate, ImagesFromList
from scipy.io import loadmat
from tqdm import tqdm
from .augmentor.image_augmentor import DataAugmentor
from utils.common_utils import reFormatDict, load_to_gpu

from PIL import Image
import torch.utils.data as data
import numpy as np
import math
import random
import torch

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
        self.pretext_task = config.PRETEXT_TASK
        self.bs = config.TRAINING.BATCH_SIZE
        self.bs_cluster = config.TRAINING.BATCH_SIZE_CLUSTER
        self.workers = config.TRAINING.WORKERS
        
        # 在pittbugrh数据集中各个weight相等
        self.dbStruct = self.parse_dbStruct(join(self.root_path, config.PATH.MAT_PATH))

        # 分别导入db和q的images
        self.dbImages = np.asarray([join(self.root_path, dbIm) for dbIm in self.dbStruct.dbImage], dtype=object)
        queries_dir = self.root_path + config.PATH.QUERIES_PATH
        self.qImages = np.asarray([join(queries_dir, qIm) for qIm in self.dbStruct.qImage], dtype=object)
        self.allImages = np.concatenate((self.dbImages,self.qImages),axis=0)
        
        # 为了满足msls格式的dataloader，实际上表示城市，这里pitts只有一个城市，即一个元素
        self.qEndPosList, self.dbEndPosList = [len(self.qImages)], [len(self.dbImages)]

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

    def __len__(self):
        return len(self.triplets)

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
    
    def collate_fn(self, batch):
        
        data_dict = reFormatDict(batch)
        ret = {}
        for key, val in data_dict.items():
            try:
                if key in ['nQuery', 'nPos', 'nNeg']:
                    ret[key] = np.vstack(val).transpose(0, 3, 1, 2)
                elif key in ['negUse']:
                    ret[key] = np.concatenate(val)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError
        
        ret['image'] = np.vstack((ret['nQuery'], ret['nPos'], ret['nNeg']))
        [ret.pop(x) for x in ['nQuery', 'nPos', 'nNeg']]
        return ret

    def prepare_data(self, neg, query, pos):
        allImages = np.concatenate((query, pos, neg), 0)
        temp = {'image': allImages}
        imageAug = self.augmentor.forward(temp)
        return imageAug['image']

    def new_epoch(self):

        # refresh ways to get
        self.nCacheSubset = math.ceil(len(self.qIdx) / self.cached_queries)
        arr = np.arange(len(self.qIdx))
        arr = random.choices(arr, self.weights, k=len(arr))
        self.subcache_indices = np.array_split(arr, self.nCacheSubset)

    def update_subcache(self, model):

        opt = {'batch_size': self.bs_cluster, 'shuffle': False, 'num_workers': self.workers, \
                        'pin_memory': True, 'collate_fn': ImagesFromList.collect_fn_img_load}
        out_dim = model.feature_selecter.num_clusters * model.backbone.out_dim
        model.eval()
        with torch.no_grad():
            allvecs = torch.zeros(len(self.allImages), out_dim).to(self.device)
            big_loader = torch.utils.data.DataLoader(ImagesFromList(self.allImages, transform=self.augmentor), **opt)
            for i, data_dict in tqdm(enumerate(big_loader), desc='compute all descriptors', total=len(self.allImages) // self.bs_cluster,
                                    position=2, leave=False):
                    load_to_gpu(data_dict)
                    batch_dict = model(data_dict)
                    allvecs[i * self.bs_cluster:(i + 1) * self.bs_cluster, :] = batch_dict['clustered_feature']
            allvecs = allvecs.cpu().numpy()
        # 若过大，则应考虑写入硬盘
        self.allvecs = allvecs

    def __getitem__(self, idx):
        
        ret = {}
        # get triplet

        
        
        
