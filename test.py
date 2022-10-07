import torch
import faiss
import numpy as np

from torch.utils.data import DataLoader
from datasets.dataset import ImagesFromList
from tqdm import tqdm
from utils.common_utils import load_to_gpu


def test_model(model, val_dataset, threads, cuda, device, pbar_position=0):

    eval_set_queries = ImagesFromList(images=val_dataset.qImages, transform=val_dataset.augmentor)
    eval_set_dbs = ImagesFromList(images=val_dataset.dbImages, transform=val_dataset.augmentor)
    
    test_data_loader_queries = DataLoader(dataset=eval_set_queries, num_workers=threads, batch_size= val_dataset.bs,
                                          shuffle=False, pin_memory=cuda, collate_fn=eval_set_queries.collect_fn_img_load)
    test_data_loader_dbs = DataLoader(dataset=eval_set_dbs, num_workers=threads, batch_size= val_dataset.bs,
                                      shuffle=False, pin_memory=cuda, collate_fn=eval_set_queries.collect_fn_img_load)
                              
    model.eval()
    with torch.no_grad():
        tqdm.write('====> Extracting Features')
        pool_size = model.backbone.out_dim * model.feature_selecter.num_clusters
        qFeat = np.empty((len(eval_set_queries), pool_size), dtype=np.float32)
        dbFeat = np.empty((len(eval_set_dbs), pool_size), dtype=np.float32)

        for feat, test_data_loader in zip([qFeat, dbFeat], [test_data_loader_queries, test_data_loader_dbs]):
            # 对于val set的loader中的db和q分别遍历
            for iteration, data_dict in enumerate(tqdm(test_data_loader, position=pbar_position, leave=False, desc='Test Iter'.rjust(15)), 1):
                
                load_to_gpu(data_dict)
                indices = data_dict['idx']
                data_dict = model(data_dict)
                vlad_encoding = data_dict['clustered_feature']
                feat[indices.detach().cpu().numpy()[:, 0], :] = vlad_encoding.detach().cpu().numpy()

    del test_data_loader_queries, test_data_loader_dbs
    tqdm.write('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(pool_size)
    # noinspection PyArgumentList
    faiss_index.add(dbFeat)

    tqdm.write('====> Calculating recall @ N')
    n_values = [1, 5, 10, 20, 50, 100]

    # for each query get those within threshold distance
    gt = val_dataset.pIdx

    # any combination of mapillary cities will work as a val set
    qEndPosTot = 0
    dbEndPosTot = 0
    for cityNum, (qEndPos, dbEndPos) in enumerate(zip(val_dataset.qEndPosList, val_dataset.dbEndPosList)):
        faiss_index = faiss.IndexFlatL2(pool_size)
        faiss_index.add(dbFeat[dbEndPosTot:dbEndPosTot+dbEndPos, :])
        _, preds = faiss_index.search(qFeat[qEndPosTot:qEndPosTot+qEndPos, :], max(n_values))
        if cityNum == 0:
            predictions = preds
        else:
            predictions = np.vstack((predictions, preds))
        qEndPosTot += qEndPos
        dbEndPosTot += dbEndPos

    correct_at_n = np.zeros(len(n_values))
    # TODO can we do this on the matrix in one go?
    for qIx, pred in enumerate(predictions):
        for i, n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / len(val_dataset.qIdx)

    all_recalls = {}  # make dict for output
    for i, n in enumerate(n_values):
        all_recalls[n] = recall_at_n[i]
        tqdm.write("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
        
    return all_recalls

