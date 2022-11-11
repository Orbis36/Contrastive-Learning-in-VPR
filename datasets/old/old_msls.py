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

def ori_get_item():
    triplet, target = self.triplets[idx]

        # get query, positive and negative idx
        qidx = triplet[0]
        pidx = triplet[1]

        # load images into triplet list
        pImage = np.array(Image.open(self.dbImages[pidx]))[np.newaxis, ...]
        qImage = np.array(Image.open(self.qImages[qidx]))[np.newaxis, ...]

        # load neg samples
        num_use = sum(np.array(target) == 0)
        nidx = triplet[2:]
        negImage = np.stack([np.array(Image.open(self.dbImages[idx])) for idx in nidx], axis=0)

        imageAug = self.prepare_data(negImage, qImage, pImage)
        ret['nQuery'] = imageAug[0:1, ...]
        ret['nPos'] = imageAug[1:2, ...]
        ret['nNeg'] = imageAug[2:, ...]
        ret['negUse'] = np.array(num_use).reshape(-1, 1)
        return ret

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
        opt = {'batch_size': self.bs_cluster, 'shuffle': False, 'num_workers': self.workers, 'pin_memory': True, 'collate_fn': ImagesFromList.collect_fn_img_load}
        qloader = torch.utils.data.DataLoader(ImagesFromList(self.qImages[qidxs], transform=self.augmentor), **opt)
        dbloader = torch.utils.data.DataLoader(ImagesFromList(self.dbImages, transform=self.augmentor), **opt)

        # TODO: 换用另一种KNN的选择方式
        # calculate their descriptors
        out_dim = net.feature_selecter.num_clusters * net.backbone.out_dim

        """
        net.eval()
        with torch.no_grad():
            # initialize descriptors
            out_dim = net.feature_selecter.num_clusters * net.backbone.out_dim
            qvecs = torch.zeros(len(qidxs), out_dim).to(self.device)
            dbvecs = torch.zeros(len(self.dbImages), out_dim).to(self.device)

            bs = opt['batch_size']
            model_weight = net.state_dict()
            # compute descriptors
            for i, data_dict in tqdm(enumerate(dbloader), desc='compute database descriptors', total=len(self.dbImages) // self.bs_cluster,
                                 position=2, leave=False):
                load_to_gpu(data_dict)
                batch_dict = net(data_dict)
                dbvecs[i * bs:(i + 1) * bs, :] = batch_dict['clustered_feature']

            for i, data_dict in tqdm(enumerate(qloader), desc='compute query descriptors', total=len(qidxs) // self.bs_cluster,
                                 position=2, leave=False):
                load_to_gpu(data_dict)
                batch_dict = net(data_dict)
                qvecs[i * bs:(i + 1) * bs, :] = batch_dict['clustered_feature']
        """
        # 载入事先做好的cache
        # 第二次时用net做cache
        if self.FIRST: 
            h5 = h5py.File('./train_feat_cache.hdf5', mode='r')
            allvecs = h5.get("features")
            self.FIRST = False
        else:
            net.eval()
            with torch.no_grad():
                allvecs = torch.zeros(len(self.allImages), out_dim).to(self.device)
                big_loader = torch.utils.data.DataLoader(ImagesFromList(self.allImages, transform=self.augmentor), **opt)
                for i, data_dict in tqdm(enumerate(big_loader), desc='compute all descriptors', total=len(self.allImages) // self.bs_cluster,
                                        position=2, leave=False):
                        load_to_gpu(data_dict)
                        batch_dict = net(data_dict)
                        allvecs[i * self.bs_cluster:(i + 1) * self.bs_cluster, :] = batch_dict['clustered_feature']
                allvecs = allvecs.cpu().numpy()
        tqdm.write('>> Searching for hard negatives...')
        tqdm.write('>> Fitting Data')

        # 将相似度搜索改为knn形式
        # selection of hard triplets
        for q in tqdm(range(len(qidxs)), desc='Hard Negative mining', total=len(qidxs), position=2, leave=False):
            # 拿到当前的query在所有qidxs里的index
            qidx_all = qidxs[q]
            qFeat = allvecs[10000+qidx_all]

            # 如果没有pos，跳过这个sample
            if len(self.pIdx[qidx_all]) == 0:
                continue
            
            # positive 和 negative都是在all里取
            knn_ps = NearestNeighbors(n_jobs=-1) # TODO replace with faiss?
            pos_sample = allvecs[self.pIdx[qidx_all]]
            knn_ps.fit(pos_sample)
            dPos, posNN = knn_ps.kneighbors(qFeat.reshape(1,-1), 1)
            dPos = dPos.item()
            need_pos_idx = self.pIdx[qidx_all][posNN[0]]

            knn_ng = NearestNeighbors(n_jobs=-1) # TODO replace with faiss?
            neg_sample_idx = np.unique(np.random.choice(self.nonNegIdx[qidx_all], 1000))
            neg_sample = allvecs[neg_sample_idx]

            knn_ng.fit(neg_sample)
            dNeg, negNN = knn_ng.kneighbors(qFeat.reshape(1,-1), self.nNeg*10)
            dNeg = dNeg.reshape(-1)
            negNN = negNN.reshape(-1)

            violatingNeg = dNeg < dPos + self.margin**0.5

            if np.sum(violatingNeg) < 1:
                continue

            negNN = list(negNN[violatingNeg][:self.nNeg])
            # package the triplet and target
            triplet = [qidx_all, need_pos_idx[0], *negNN]
            target = [-1, 1] + [0] * len(negNN)

            if self.nNeg - len(negNN) > 0:
                repeat = self.nNeg - len(negNN)
                target = target + [-2]*repeat
            self.triplets.append((triplet, target))

        # increment subset counter
        self.current_subset += 1

