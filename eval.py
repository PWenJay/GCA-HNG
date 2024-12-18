import os
import torch
import argparse
from tqdm import tqdm
from scripts.eval_utils import *
from config import cfg, cfg_from_file
from scripts.dataset import MetricDataset, Inshop_Dataset
import scripts.models as models


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', required=True, help='Config file for training (and optionally testing)')
    parser.add_argument('--model_path', required=True, help='dataset name')
    return parser.parse_args()


def eval_model(cfg, opt, batch_size=128, workers=8):
    device = "cuda"

    if 'SOP' in cfg.TEST.DATASET:
        classes = 11318
    elif 'CAR' in cfg.TEST.DATASET:
        classes = 98
    elif 'CUB' in cfg.TEST.DATASET:
        classes = 100
    elif 'IN_SHOP' in cfg.TEST.DATASET:
        classes = 3997

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
        
        model = models.GCA_HNG(cfg, embedding_size=cfg.MODEL.EMBEDDING_SIZE, n_class=classes, pretrained=cfg.MODEL.PRETRAINED, num_per_class=cfg.TRAIN.NUM_PER_CLASS, is_norm=cfg.MODEL.NORM, is_pooling=cfg.MODEL.POOLING, bn_freeze=cfg.SOLVER.BN_FREEZE, beta=cfg.LOSS.BETA, gnn_layers=cfg.MODEL.GNN_LAYER, softmax_factor=cfg.LOSS.SOFTMAX_FACTOR).to(device)
        model.load_state_dict(torch.load(opt.model_path, map_location="cuda"), False)
        model.eval()

        embedding_list = []
        label_list = []
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
            evaluate_retrieval(X, T, neighbours=[1, 2, 4, 8])
        else:
            evaluate_retrieval(X, T, neighbours=[1, 10, 100, 1000])
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

        model = models.GCA_HNG(cfg, embedding_size=cfg.MODEL.EMBEDDING_SIZE, n_class=classes, pretrained=cfg.MODEL.PRETRAINED, num_per_class=cfg.TRAIN.NUM_PER_CLASS, is_norm=cfg.MODEL.NORM, is_pooling=cfg.MODEL.POOLING, bn_freeze=cfg.SOLVER.BN_FREEZE, beta=cfg.LOSS.BETA, gnn_layers=cfg.MODEL.GNN_LAYER, softmax_factor=cfg.LOSS.SOFTMAX_FACTOR).to(device)
        model.load_state_dict(torch.load(opt.model_path, map_location="cuda"), False)
        model.eval()

        evaluate_retrieval_Inshop(model, dl_query, dl_gallery)


if __name__ == "__main__":
    opt = parse_args()
    cfg_from_file(opt.cfg_file)
    eval_model(cfg, opt)
