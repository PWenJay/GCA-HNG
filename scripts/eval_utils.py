import math
import numpy as np
from tqdm import tqdm
from scipy.special import comb
from sklearn import cluster
from sklearn import neighbors
import torch
import torch.nn.functional as F
from scripts.utils import l2_norm


def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    s = 0
    for t,y in zip(T,Y):
        if t in torch.Tensor(y).long()[:k]:
            s += 1
    return s / (1. * len(T))


def evaluate_retrieval(X,T,neighbours):
    cls_num = torch.bincount(T)[T]
    ap_atR = []
    r_p=[]

    X = l2_norm(X)
    K = 1000
    Y = []
    xs = []
    T = T.cuda()
    for x in X:
        if len(xs)<10000:
            xs.append(x)
        else:
            xs.append(x)            
            xs = torch.stack(xs,dim=0)
            cos_sim = F.linear(xs,X)
            y = T[cos_sim.topk(1 + K)[1][:,1:]]
            Y.append(y.float().cpu())
            xs = []
            
    # Last Loop
    xs = torch.stack(xs,dim=0)
    cos_sim = F.linear(xs,X)
    y = T[cos_sim.topk(1 + K)[1][:,1:]]
    Y.append(y.float().cpu())
    Y = torch.cat(Y, dim=0)
    T = T.detach().cpu()


    # calculate recall @ 1, 2, 4, 8
    recall = []
    for k in neighbours:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))

    y = np.array(Y)
    t = np.array(T)
    for topk, gt, R in zip(y, t, cls_num):
        R = R - 1
        map_labels = torch.tensor(np.isin(topk, gt), dtype=torch.uint8)
        r_p.append(torch.sum(map_labels[:R], dim=0) / R)
        precisions = torch.cumsum(map_labels, dim=0) * map_labels # Consider only positions corresponding to GTs
        precisions = precisions/torch.arange(1, map_labels.shape[0] + 1)
        ap_atR.append(float(torch.sum(precisions[:R]) / R))

    map_atR = np.mean(ap_atR) * 100
    mr_p = np.mean(r_p) * 100
    print("map@R:{:.3f}".format(map_atR))
    print("r_p:{:.3f}".format(mr_p))
    return recall, map_atR, mr_p


def predict_batchwise(model, dataloader):
    device = "cuda"
    model_is_training = model.training
    model.eval()
    
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    with torch.no_grad():
                        J = model.feature_extractor(J.to(device))

                for j in J:
                    A[i].append(j)
    model.train()
    model.train(model_is_training) # revert to previous training state
    
    return [torch.stack(A[i]) for i in range(len(A))]


def evaluate_map_Inshop(cos_sim, query_T, gallery_T):
    ap_atR = []
    r_p = []
    K = 1000
    cls_num = torch.bincount(gallery_T)[query_T]
    y = gallery_T[cos_sim.topk(K)[1]]
    for topk, gt, R in zip(y, query_T, cls_num):
        map_labels = torch.tensor(np.isin(topk, gt), dtype=torch.uint8)
        r_p.append(torch.sum(map_labels[:R], dim=0) / R)
        precisions = torch.cumsum(map_labels, dim=0) * map_labels # Consider only positions corresponding to GTs
        precisions = precisions/torch.arange(1, map_labels.shape[0] + 1)
        ap_atR.append(float(torch.sum(precisions[:R]) / R))
    
    map_atR = np.mean(ap_atR) * 100
    mr_p = np.mean(r_p) * 100
    print("map@R:{:.3f}".format(map_atR))
    print("r_p:{:.3f}".format(mr_p))

    return map_atR, mr_p

def evaluate_retrieval_Inshop(model, query_dataloader, gallery_dataloader):
    # calculate embeddings with model and get targets
    query_X, query_T = predict_batchwise(model, query_dataloader)
    gallery_X, gallery_T = predict_batchwise(model, gallery_dataloader)
    
    query_X = l2_norm(query_X)
    gallery_X = l2_norm(gallery_X)
    cos_sim = F.linear(query_X, gallery_X)

    def recall_k(cos_sim, query_T, gallery_T, k):
        m = len(cos_sim)
        match_counter = 0

        for i in range(m):
            pos_sim = cos_sim[i][gallery_T == query_T[i]]
            neg_sim = cos_sim[i][gallery_T != query_T[i]]

            thresh = torch.max(pos_sim).item()

            if torch.sum(neg_sim > thresh) < k:
                match_counter += 1
            
        return match_counter / m
    
    # calculate recall @ 1, 2, 4, 8
    recall = []
    for k in [1, 10, 20, 30, 40, 50]:
        r_at_k = recall_k(cos_sim, query_T, gallery_T, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))

    map_atR, R_p = evaluate_map_Inshop(cos_sim, query_T, gallery_T)     
    return recall, map_atR, R_p
