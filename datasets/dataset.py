import torch.utils.data as torch_data
import numpy as np
from collections import namedtuple
from sklearn.neighbors import NearestNeighbors

class DatasetTemplate(torch_data.Dataset):
    def __init__(self, config, mode):
        super().__init__()
        self.training = mode
        # number of negatives to randomly sample
        self.nNegSample = config.SAMPLE.N_NEG_SAMPLE
        # number of negatives used for training
        self.nNegUse = config.SAMPLE.NEG_USE
        self.dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset', 
        'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
        'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])

    def query_preprocess(self, dbStruct):
        # TODO: Faiss accelerated need to be implemented.
        # If you need use different pretext task, implement the method in sub-class
        
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(dbStruct.utmDb)

        # Find the obvious positive sample and sort them by distance
        nontrivial_positives = list(knn.radius_neighbors(dbStruct.utmQ,
            radius=dbStruct.nonTrivPosDistSqThr**0.5, 
            return_distance=False))
        for i,posi in enumerate(nontrivial_positives):
            nontrivial_positives[i] = np.sort(posi)

        # Filter the queries that doesn't have non trivial potential positives
        queries = np.where(np.array([len(x) for x in nontrivial_positives])>0)[0]
        
        # potential negatives are those outside of posDistThr range
        potential_positives = knn.radius_neighbors(dbStruct.utmQ,
                radius=dbStruct.posDistThr, 
                return_distance=False)

        potential_negatives = []
        for pos in potential_positives:
            potential_negatives.append(np.setdiff1d(np.arange(dbStruct.numDb),
                pos, assume_unique=True))
        
        return potential_negatives, queries, nontrivial_positives

            

    @property
    def check_mode(self):
        return 'train' if self.training else 'test'

    def __getitem__(self, index):
        raise NotImplementedError

    def update_subcache(self):
        pass
