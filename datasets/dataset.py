import torch.utils.data as torch_data
import numpy as np
import torch
import h5py

from collections import namedtuple
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from PIL import Image
from utils.common_utils import load_to_gpu, reFormatDict
    
# A mini dataset class for image loading in desc. calculation
class ImagesFromList(torch_data.Dataset):
    def __init__(self, images, transform):
        self.images = np.asarray(images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        imgs = Image.open(self.images[idx])
        imgs = np.array(imgs, dtype=np.float32)
        data_dict = {'image': imgs, 'idx': idx}
        data_dict = self.transform.forward(data_dict=data_dict)
        return data_dict
    
    @staticmethod
    def collect_fn_img_load(batch_list):

        data_dict = reFormatDict(batch_list=batch_list)
        ret = {}
        for key, val in data_dict.items():
            try:
                if key in ['image']:
                    ret[key] = np.stack(val, axis=0).transpose(0, 3, 1, 2)
                if key in ['idx']:
                    ret[key] = np.array(val).reshape(-1, 1)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError
        return ret


class DatasetTemplate(torch_data.Dataset):
    def __init__(self, config, mode):
        super().__init__()
        self.training = mode
        # number of negatives to randomly sample
        self.nNegSample_times = config.SAMPLE.N_NEG_SAMPLE_TIMES
        self.cached_queries = config.SAMPLE.CACHE_SAMPLE
        self.cached_negatives = self.cached_queries * self.nNegSample_times
        self.FIRST = True

        # samples which is 100% negative
        self.nonNeg = config.SAMPLE.NEG_DIST

        # number of negatives used for training
        self.nNeg = config.SAMPLE.NEG_USE
        self.dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset', 
        'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
        'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])

    def query_preprocess(self, dbStruct):
        # TODO: Faiss accelerated need to be implemented.
        # If you need use different pretext task, implement the method in sub-class
        
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(dbStruct.utmDb)

        pos_radius = dbStruct.nonTrivPosDistSqThr**0.5 if self.training == 'train' else 25
        # Find the obvious positive sample and sort them by distance
        nontrivial_positives = list(knn.radius_neighbors(dbStruct.utmQ,
            radius=pos_radius, 
            return_distance=False))

        for i,posi in enumerate(nontrivial_positives):
            nontrivial_positives[i] = np.sort(posi)

        # Filter the queries that doesn't have non trivial potential positives
        queries = np.where(np.array([len(x) for x in nontrivial_positives])>0)[0]
        a = nontrivial_positives[3456]
        for idx, x in enumerate(queries):
            if idx != x:
                print()
        
        # absolute negatives are those outside of posDistThr range
        potential_positives = knn.radius_neighbors(dbStruct.utmQ,
                radius=self.nonNeg, 
                return_distance=False)

        nontrival_negatives = []
        for pos in potential_positives:
            nontrival_negatives.append(np.setdiff1d(np.arange(dbStruct.numDb),
                pos, assume_unique=True))
        
        return np.asarray(nontrival_negatives, dtype=object), queries, np.asarray(nontrivial_positives, dtype=object)

    @property
    def check_mode(self):
        return 'train' if self.training else 'test'

    def __getitem__(self, index):
        raise NotImplementedError

    