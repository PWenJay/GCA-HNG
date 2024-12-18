
import os
import time
import shutil
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim

from scripts import models
from scripts.dataset import MetricDataset, Inshop_Dataset, BalancedBatchSampler
from scripts.evaluate import evaluation
from tensorboardX import SummaryWriter

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from config import cfg, cfg_from_file


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', required=True, help='Config file for training (and optionally testing)')
    return parser.parse_args()


class AverageMeter(object):

	def __init__(self, init):
		self.avg = init
		self.arr = []

	def reset(self):
		self.avg = np.mean(self.arr)
		self.arr = []

	def update(self, val, n=1):
		self.arr.append(val)


def trainer(cfg, cfg_path):
    device = "cuda"
    # data prepare
    model_path = cfg.TRAIN.MODEL_PATH
    data_path = cfg.TRAIN.DATASET
    if data_path != 'data/IN_SHOP/class/train':
        mdata = MetricDataset(cfg.TRAIN.DATASET, scale=cfg.TRAIN.SCALE)
    else:
        mdata = Inshop_Dataset(root='data', mode='train')
    num_per_class = cfg.TRAIN.NUM_PER_CLASS
    sampled_class_num = cfg.TRAIN.NUM_SAMPLED_CLASSES
    batch_size = sampled_class_num * num_per_class
    sampler = BalancedBatchSampler(mdata.targets, n_classes=sampled_class_num, n_samples=num_per_class)
    train_loader = torch.utils.data.DataLoader(
        mdata,  
        batch_sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
    )

    # tensorboard init
    save_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    print("expriment running at:", save_time)
    tensorboard_path = os.path.join(cfg.TRAIN.BOARD_LOG_PATH, opt.cfg_file.split('/')[-1].split('.')[0], save_time)
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    shutil.copyfile(cfg_path, os.path.join(tensorboard_path, cfg_path.split('/')[-1]))
    try:
        summary_writer = SummaryWriter(tensorboard_path, comment="hng")
        print('log tensorboard summaty at {}'.format(tensorboard_path))
    except:
        summary_writer = None
        print('Can not use tensorboard')

    # model init
    embedding_size = cfg.MODEL.EMBEDDING_SIZE
    pretrained = cfg.MODEL.PRETRAINED
    is_norm = cfg.MODEL.NORM
    is_pooling=cfg.MODEL.POOLING
    bn_freeze = cfg.SOLVER.BN_FREEZE
    patch_freeze = cfg.SOLVER.PATCH_FREEZE
    
    model = models.GCA_HNG(cfg, embedding_size=embedding_size, n_class=len(mdata.classes), pretrained=pretrained, num_per_class=num_per_class, is_norm=is_norm,is_pooling=is_pooling, bn_freeze=bn_freeze, beta=cfg.LOSS.BETA, gnn_layers=cfg.MODEL.GNN_LAYER, softmax_factor=cfg.LOSS.SOFTMAX_FACTOR).to(device)
   
    # optim init
    epoches = cfg.TRAIN.EPOCHES
    lr_metric = cfg.SOLVER.BASE_LR
    metirc_gamma = cfg.SOLVER.METRIC_GAMMA
    lr_s = cfg.SOLVER.CLASSIFIER_LR

    optimizer_general = optim.AdamW
    optimizer_s = optimizer_general(model.softmax_classifier.parameters(), lr=lr_s)

    lr_gnn = cfg.SOLVER.GNN_LR
    gnn_gamma = cfg.SOLVER.GNN_GAMMA
    optimizer_gnn = optimizer_general(model.gnn.parameters(), lr=lr_gnn, weight_decay=cfg.SOLVER.WEIGHT_DECAY)

    metric_model_param = [{'params':model.feature_extractor.backbone.parameters(), 'lr':lr_metric}, 
                            {'params':model.feature_extractor.fc.parameters(), 'lr': lr_metric * cfg.SOLVER.EMBEDDING_RATIO}]
    optimizer_c = optimizer_general(metric_model_param, lr=lr_metric, weight_decay=cfg.SOLVER.WEIGHT_DECAY)

    if cfg.SOLVER.LR_POLICY == 'step':
        scheduler_c = optim.lr_scheduler.StepLR(optimizer_c, step_size=cfg.SOLVER.METRIC_DECAY_STEPS, gamma=metirc_gamma)
    elif cfg.SOLVER.LR_POLICY == 'cos':
        scheduler_c = optim.lr_scheduler.CosineAnnealingLR(optimizer_c, cfg.SOLVER.METRIC_DECAY_STEPS, eta_min=0, last_epoch=-1)
    elif cfg.SOLVER.LR_POLICY == 'exp':
        scheduler_c = optim.lr_scheduler.ExponentialLR(optimizer_c, gamma=0.9)
    else:
        scheduler_c = None

    if optimizer_gnn is not None:
        scheduler_gnn = optim.lr_scheduler.CosineAnnealingLR(optimizer_gnn, float(epoches))
    else:
        scheduler_gnn = None

    # start training
    J_avg = AverageMeter(1e6)
    J_gen = AverageMeter(1e6)
    recalls_dict = {}
    map_dict = {}
    Rp_dict = {}
    best_r1 = 0
    best_result = {}
    iters_per_epoch = len(mdata)//(batch_size)

    for epoch in range(epoches):
        model.train()    
        jm = 1e6
        jgen = 1e6
        
        if bn_freeze:
            modules = model.feature_extractor.modules()
            for m in modules: 
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()

        if patch_freeze:
            modules = model.feature_extractor.named_modules()
            for name, m in modules:
                if 'patch_embed' in name:
                    m.eval()

        if cfg.SOLVER.WARM_UP_EPOCH > 0:
            freeze_model_param = set(model.feature_extractor.parameters()).difference(set(model.feature_extractor.fc.parameters()))
            if epoch < cfg.SOLVER.WARM_UP_EPOCH:
                for param in list(freeze_model_param):
                    param.requires_grad = False
            else:
               for param in list(freeze_model_param):
                    param.requires_grad = True

        with tqdm(total=iters_per_epoch, desc=f'Epoch {epoch + 1}/{epoches}', ncols=130, postfix=dict, mininterval=0.3) as pbar:
            for i, batch in enumerate(train_loader):
                x_batch, label = batch
                x_list = []
                label_list = []
                for id in range(num_per_class):
                    x_list.append(x_batch[id::num_per_class])
                    label_list.append(label[id::num_per_class])
                x_batch = torch.cat(x_list, dim=0)
                label = torch.cat(label_list, dim=0)

                jgen, jmetric, jm, ce, e_bj, metrics = model(x_batch.to(device), label.to(device), J_avg.avg, J_gen.avg, optimizer_c, optimizer_s, optimizer_gnn) 
                jgen = max(jgen.item(), 1.0e-6)
                J_gen.update(jgen)
                
                jm = max(jm.item(), 1.0e-6)   
                J_avg.update(jm)

                pbar.set_postfix(**{"Jmetric": jmetric.item(), "Jgen": jgen, "Jm": jm, "CrossEntropy": ce.item(), "e_bj":e_bj}) 
                pbar.update(1)
                summary_writer.add_scalars('hng/losses', metrics, epoch * (len(mdata) // batch_size) + i)           

        if (epoch+1) % cfg.TEST.EVAL_EPOCH == 0:
            if cfg.TEST.DATASET == 'data/SOP/class/test':
                recalls, map_R, R_p = evaluation(cfg, model)
                for i, k in enumerate([1, 10, 100, 1000]):
                    recalls_dict.update({str(k):float(recalls[i])})
            elif cfg.TEST.DATASET == 'data/IN_SHOP/class/test':
                recalls, map_R, R_p = evaluation(cfg, model)
                for i, k in enumerate([1, 10, 20, 30, 40, 50]):
                    recalls_dict.update({str(k):float(recalls[i])})
            else:
                recalls, map_R, R_p = evaluation(cfg, model)
                for i, k in enumerate([1, 2, 4, 8]):
                    recalls_dict.update({str(k):float(recalls[i])})
        
            map_dict.update({"map_R": round(map_R, 3)})
            Rp_dict.update({"R_p": round(R_p, 3)})

            summary_writer.add_scalars('hsg/recalls', recalls_dict, 10 * epoch * (len(mdata) // batch_size) + i)
            summary_writer.add_scalars('hsg/map_R', map_dict, 10 * epoch * (len(mdata) // batch_size) + i)
            summary_writer.add_scalars('hsg/R_p', Rp_dict, 10 * epoch * (len(mdata) // batch_size) + i)


            if not os.path.exists(os.path.join(model_path, opt.cfg_file.split('/')[-1].split('.')[0], save_time)):
                os.makedirs(os.path.join(model_path, opt.cfg_file.split('/')[-1].split('.')[0], save_time))
            
            if recalls[0] > best_r1:
                best_r1 = recalls[0]
                best_result['recall'] = recalls_dict.copy()
                best_result['map_R'] = map_dict.copy()
                best_result['R_p'] = Rp_dict.copy()
                torch.save(model.state_dict(), os.path.join(model_path, opt.cfg_file.split('/')[-1].split('.')[0], save_time, 'bestR_1.pth'))

            # torch.save(model.state_dict(), os.path.join(model_path, opt.cfg_file.split('/')[-1].split('.')[0], save_time, 'epoch_%d.pth' % (epoch+1)))
        J_avg.reset()
        J_gen.reset()
        if scheduler_c is not None:
            scheduler_c.step()
        scheduler_gnn.step()

    with open('logs/' + opt.cfg_file.split('/')[-1][:-4] + '_log.txt', 'a') as f:
        f.write(opt.cfg_file.split('/')[-1] + '\n' +  save_time + '\t' + str(best_result) + '\n')
        f.write('_'*30 + '\n')

if __name__ == "__main__":
    opt = parse_args()
    cfg_from_file(opt.cfg_file)
    trainer(cfg, opt.cfg_file)
