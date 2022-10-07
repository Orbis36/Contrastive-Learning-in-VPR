from .model_template import VPRModelTemplate
from math import ceil
from torch.utils.data import DataLoader, SubsetRandomSampler
from os.path import join
from datasets.dataset import ImagesFromList
from tqdm.auto import tqdm
from utils.common_utils import load_to_gpu

import torch.nn.functional as F
import numpy as np
import h5py
import torch
import faiss
import os


class NetVLAD(VPRModelTemplate):
    def __init__(self, model_cfg, cache_dataset):
        super().__init__(model_cfg)
        self.module_list = self.build_networks(model_cfg=model_cfg)
        self.create_model_dir(self.__class__.__name__)

        self.cacheDescsNum = model_cfg.CACHE.DESC_NUM
        self.cacheDescPerImg = model_cfg.CACHE.DESC_PER_IMG
        self.nImg = int(self.cacheDescsNum / self.cacheDescPerImg)

        if model_cfg.FEATURE_SELECT.INIT_PARAM:
            self.get_cluster(cache_dataset)

        with h5py.File(self.initcache_clusters_path, mode='r') as h5:
            clsts = h5.get("centroids")[...]
            traindescs = h5.get("descriptors")[...]
            self.feature_selecter.init_params(clsts, traindescs)
            del clsts, traindescs


    def forward(self, batch_dict, need_grad=True):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        return batch_dict

    def get_training_loss(self, batch_dict):
        loss_triplet = self.feature_selecter.get_feature_selected_loss(batch_dict)
        return loss_triplet

    def get_cluster(self, cache_dataset):
        cluster_sampler = SubsetRandomSampler(np.random.choice(len(cache_dataset.dbImages), self.nImg, replace=False))
        cluster_data_loader = DataLoader(dataset=ImagesFromList(cache_dataset.dbImages, transform=cache_dataset.augmentor),
                                        num_workers=cache_dataset.workers, batch_size=cache_dataset.bs, shuffle=False,
                                        pin_memory=True, sampler=cluster_sampler, collate_fn=ImagesFromList.collect_fn_img_load)
        file_name = self.backbone.encoder_name + '_' + cache_dataset.split + '_' + str(self.feature_selecter.num_clusters) + '_desc_cen.hdf5'
        self.initcache_clusters_path = join(self.cache_save_path, file_name)
        # encoder_dim = self.backbone.out_dim * self.feature_selecter.num_clusters
        
        with h5py.File(self.initcache_clusters_path, mode='w') as h5_file:
            with torch.no_grad():
                self.eval()
                self.cuda()
                tqdm.write('====> Extracting Descriptors')
                dbFeat = h5_file.create_dataset("descriptors", [self.cacheDescsNum, self.backbone.out_dim], dtype=np.float32)
                
                for iteration, batch_dict in enumerate(tqdm(cluster_data_loader, desc='Iter'.rjust(15)), 1):
                    load_to_gpu(batch_dict=batch_dict)
                    batch_dict = self.forward(batch_dict)
                    feature_from_backbone = batch_dict['feature_map'].view(cache_dataset.bs, self.backbone.out_dim, -1).permute(0, 2, 1)
                    image_descriptors = F.normalize(feature_from_backbone, p=2, dim=2) # we L2-norm descriptors before vlad so
                    # need to L2-norm here as well

                    batchix = (iteration - 1) * cache_dataset.bs * self.cacheDescPerImg
                    for ix in range(image_descriptors.size(0)):
                        # sample different location for each image in batch
                        # 相当于将backbone输出的特征按512的channel dim展开，选取100个
                        sample = np.random.choice(image_descriptors.size(1), self.cacheDescPerImg, replace=False)
                        startix = batchix + ix * self.cacheDescPerImg
                        dbFeat[startix:startix + self.cacheDescPerImg, :] = image_descriptors[ix, sample, :].detach().cpu().numpy()
                    del batch_dict, image_descriptors

            tqdm.write('====> Clustering..')
            niter = 100
            kmeans = faiss.Kmeans(self.backbone.out_dim, self.feature_selecter.num_clusters, niter=niter, verbose=False)
            kmeans.train(dbFeat[...])

            tqdm.write('====> Storing centroids ' + str(kmeans.centroids.shape))
            h5_file.create_dataset('centroids', data=kmeans.centroids)
            tqdm.write('====> Done!')