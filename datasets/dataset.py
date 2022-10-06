import torch.utils.data as torch_data
import numpy as np
import torch

import cv2
from collections import namedtuple
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from utils.common_utils import load_to_gpu, reFormatDict
    
# A mini dataset class for image loading in desc. calculation
class ImagesFromList(torch_data.Dataset):
    def __init__(self, images, transform):
        self.images = np.asarray(images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        imgs = cv2.imread(self.images[idx])
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

        # Find the obvious positive sample and sort them by distance
        nontrivial_positives = list(knn.radius_neighbors(dbStruct.utmQ,
            radius=dbStruct.nonTrivPosDistSqThr**0.5, 
            return_distance=False))
        for i,posi in enumerate(nontrivial_positives):
            nontrivial_positives[i] = np.sort(posi)

        # Filter the queries that doesn't have non trivial potential positives
        queries = np.where(np.array([len(x) for x in nontrivial_positives])>0)[0]
        
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

    def update_subcache(self, net):
        # reset triplets
        self.triplets = []
        # take n query images, 在new_epoch中定义subcache_indice和current_subset
        if self.current_subset >= len(self.subcache_indices):
            self.current_subset = 0

        # 找到这个subset的query index和其对应的所有positive sample index
        # 注意这里的qidxs是有效query的，
        qidxs = np.array(self.subcache_indices[self.current_subset])
        pidxs = np.unique([i for idx in self.pIdx[qidxs] for i in idx])

        # 从DB里选出N倍于sub-cache大小的作为neg样本候选, and make sure that there is no positives among them
        nidxs = np.random.choice(len(self.dbImages), self.cached_negatives, replace=False)
        nidxs = nidxs[np.in1d(nidxs, np.unique([i for idx in self.nonNegIdx[qidxs] for i in idx]))]

        # make dataloaders for query, positive and negative images
        opt = {'batch_size': self.bs, 'shuffle': False, 'num_workers': self.workers, 'pin_memory': True, 'collate_fn': ImagesFromList.collect_fn_img_load}
        qloader = torch.utils.data.DataLoader(ImagesFromList(self.qImages[qidxs], transform=self.augmentor), **opt)
        ploader = torch.utils.data.DataLoader(ImagesFromList(self.dbImages[pidxs], transform=self.augmentor), **opt)
        nloader = torch.utils.data.DataLoader(ImagesFromList(self.dbImages[nidxs], transform=self.augmentor), **opt)

        # calculate their descriptors
        net.eval()
        with torch.no_grad():
            # initialize descriptors
            out_dim = net.feature_selecter.num_clusters * net.backbone.out_dim
            qvecs = torch.zeros(len(qidxs), out_dim).to(self.device)
            pvecs = torch.zeros(len(pidxs), out_dim).to(self.device)
            nvecs = torch.zeros(len(nidxs), out_dim).to(self.device)

            bs = opt['batch_size']
            
            # compute descriptors
            for i, data_dict in tqdm(enumerate(qloader), desc='compute query descriptors', total=len(qidxs) // bs,
                                 position=2, leave=False):
                load_to_gpu(data_dict)
                batch_dict = net(data_dict)
                qvecs[i * bs:(i + 1) * bs, :] = batch_dict['clustered_feature']

            for i, data_dict in tqdm(enumerate(ploader), desc='compute positive descriptors', total=len(pidxs) // bs,
                                 position=2, leave=False):
                load_to_gpu(data_dict)
                batch_dict = net(data_dict)
                pvecs[i * bs:(i + 1) * bs, :] = batch_dict['clustered_feature']

            for i, data_dict in tqdm(enumerate(nloader), desc='compute negative descriptors', total=len(nidxs) // bs,
                                 position=2, leave=False):
                load_to_gpu(data_dict)
                batch_dict = net(data_dict)
                nvecs[i * bs:(i + 1) * bs, :] = batch_dict['clustered_feature']

        tqdm.write('>> Searching for hard negatives...')
        # compute dot product scores and ranks on GPU
        pScores = torch.mm(qvecs, pvecs.t())
        pScores, pRanks = torch.sort(pScores, dim=1, descending=True)

        # calculate distance between query and negatives
        nScores = torch.mm(qvecs, nvecs.t())
        nScores, nRanks = torch.sort(nScores, dim=1, descending=True)

        # convert to cpu and numpy
        pScores, pRanks = pScores.cpu().numpy(), pRanks.cpu().numpy()
        nScores, nRanks = nScores.cpu().numpy(), nRanks.cpu().numpy()

        # selection of hard triplets
        for q in range(len(qidxs)):

            qidx = qidxs[q]

            # find positive idx for this query (cache idx domain)
            # pidxs 是所有这个subcache里可能的pos——id
            cached_pidx = np.where(np.in1d(pidxs, self.pIdx[qidx]))

            if len(self.pIdx[qidx]) == 0:
                continue

            # find idx of positive idx in rank matrix (descending cache idx domain)
            pidx = np.where(np.in1d(pRanks[q, :], cached_pidx))

            # take the closest positve
            dPos = pScores[q, pidx][0][0]

            # get distances to all negatives
            dNeg = nScores[q, :]

            # how much are they violating
            loss = dPos - dNeg + self.margin ** 0.5
            violatingNeg = 0 < loss

            # if less than nNeg are violating then skip this query
            if np.sum(violatingNeg) <= self.nNeg:
                continue

            # select hardest negatives
            hardest_negIdx = np.argsort(loss)[:self.nNeg]

            # select the hardest negatives
            cached_hardestNeg = nRanks[q, hardest_negIdx]

            # select the closest positive (back to cache idx domain)
            cached_pidx = pRanks[q, pidx][0][0]

            # transform back to original index (back to original idx domain)
            qidx = self.qIdx[qidx]
            pidx = pidxs[cached_pidx]
            hardestNeg = nidxs[cached_hardestNeg]

            # package the triplet and target
            triplet = [qidx, pidx, *hardestNeg]
            target = [-1, 1] + [0] * len(hardestNeg)

            self.triplets.append((triplet, target))

        # increment subset counter
        self.current_subset += 1

    