import torch
from tqdm import tqdm
from scripts.eval_utils import *
from scripts.dataset import MetricDataset, Inshop_Dataset
import torch.nn.functional as F


def evaluation(cfg, model, batch_size=128, workers=8):
    device = "cuda"
    if cfg.TEST.DATASET != 'data/IN_SHOP/class/test':
        mdata = MetricDataset(cfg.TEST.DATASET, is_train=False)
        test_loader = torch.utils.data.DataLoader(
            mdata,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
            shuffle=False
        )
        model.eval()
        label_list = []
        embedding_list = []
        with tqdm(total=len(mdata)//(batch_size), ncols=130, postfix=dict, mininterval=0.3) as pbar:
            for batch in test_loader:
                x_batch, label = batch

                with torch.no_grad():
                    embedding_z = model.feature_extractor(x_batch.to(device))
                embedding_list.append(embedding_z)
                label_list.append(label)

                pbar.update(1)
        X = torch.cat(embedding_list, dim=0)
        T = torch.cat(label_list, dim=0)
        X = F.normalize(X, p=2, dim=1)

        if cfg.TEST.DATASET == 'data/CUB200/class/test' or cfg.TEST.DATASET == 'data/CARS196/class/test':
            recalls, map_R, R_p = evaluate_retrieval(X, T, neighbours=[1, 2, 4, 8])
        else:
            recalls, map_R, R_p = evaluate_retrieval(X, T, neighbours=[1, 10, 100, 1000])
        return recalls, map_R, R_p
    else:
        query_dataset = Inshop_Dataset(root='data', mode='query')
        dl_query = torch.utils.data.DataLoader(
            query_dataset,
            batch_size = 120,
            shuffle = False,
            num_workers = 8,
            pin_memory = True
        )

        gallery_dataset = Inshop_Dataset(root='data', mode='gallery')
        dl_gallery = torch.utils.data.DataLoader(
            gallery_dataset,
            batch_size = 120,
            shuffle = False,
            num_workers = 8,
            pin_memory = True
        )
        Recalls, map_R, R_p = evaluate_retrieval_Inshop(model, dl_query, dl_gallery)
        return Recalls, map_R, R_p
