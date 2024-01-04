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


def evaluate_cluster(feats, labels, n_clusters):
    """
    A function that calculate the NMI as well as F1 of a given embedding
    :param feats: The feature (embedding)
    :param labels: The labels
    :param n_clusters: How many classes
    :return: The NMI and F1 score of the given embedding
    """
    kmeans = cluster.KMeans(n_clusters=n_clusters, max_iter=100, tol=1e-6, n_init=10).fit(feats) #iter的设置
    # kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=1).fit(feats)
    centers = kmeans.cluster_centers_

    # k-nearest neighbors
    neigh = neighbors.KNeighborsClassifier(n_neighbors=1)
    neigh.fit(centers, range(len(centers)))

    labels_pred = neigh.predict(feats)
    nmi, f1 = compute_clutering_metric(labels, labels_pred)
    print('nmi: ', nmi)
    print('f1: ', f1)
    return nmi, f1


def compute_clutering_metric(idx, item_ids):
    
    N = len(idx)

    # cluster centers
    centers = np.unique(idx)
    num_cluster = len(centers)
    # print('Number of clusters: #d\n' % num_cluster);

    # count the number of objects in each cluster
    count_cluster = np.zeros(num_cluster)
    for i in range(num_cluster):
        count_cluster[i] = len(np.where(idx == centers[i])[0])

    # build a mapping from item_id to item index
    keys = np.unique(item_ids)
    num_item = len(keys)
    values = range(num_item)
    item_map = dict()
    for i in range(len(keys)):
        item_map.update([(keys[i], values[i])])

    # count the number of objects of each item
    count_item = np.zeros(num_item)
    for i in range(N):
        index = item_map[item_ids[i]]
        count_item[index] = count_item[index] + 1

    # compute purity
    purity = 0
    for i in range(num_cluster):
        member = np.where(idx == centers[i])[0]
        member_ids = item_ids[member]

        count = np.zeros(num_item)
        for j in range(len(member)):
            index = item_map[member_ids[j]]
            count[index] = count[index] + 1
        purity = purity + max(count)

    # compute Normalized Mutual Information (NMI)
    count_cross = np.zeros((num_cluster, num_item))
    for i in range(N):
        index_cluster = np.where(idx[i] == centers)[0]
        index_item = item_map[item_ids[i]]
        count_cross[index_cluster, index_item] = count_cross[index_cluster, index_item] + 1

    # mutual information
    I = 0
    for k in range(num_cluster):
        for j in range(num_item):
            if count_cross[k, j] > 0:
                s = count_cross[k, j] / N * math.log(N * count_cross[k, j] / (count_cluster[k] * count_item[j]))
                I = I + s

    # entropy
    H_cluster = 0
    for k in range(num_cluster):
        s = -count_cluster[k] / N * math.log(count_cluster[k] / float(N))
        H_cluster = H_cluster + s

    H_item = 0
    for j in range(num_item):
        s = -count_item[j] / N * math.log(count_item[j] / float(N))
        H_item = H_item + s

    NMI = 2 * I / (H_cluster + H_item)

    # compute True Positive (TP) plus False Positive (FP)
    tp_fp = 0
    for k in range(num_cluster):
        if count_cluster[k] > 1:
            tp_fp = tp_fp + comb(count_cluster[k], 2)

    # compute True Positive (TP)
    tp = 0
    for k in range(num_cluster):
        member = np.where(idx == centers[k])[0]
        member_ids = item_ids[member]

        count = np.zeros(num_item)
        for j in range(len(member)):
            index = item_map[member_ids[j]]
            count[index] = count[index] + 1

        for i in range(num_item):
            if count[i] > 1:
                tp = tp + comb(count[i], 2)

    # False Positive (FP)
    fp = tp_fp - tp

    # compute False Negative (FN)
    count = 0
    for j in range(num_item):
        if count_item[j] > 1:
            count = count + comb(count_item[j], 2)

    fn = count - tp

    # compute F measure
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    beta = 1
    F = (beta*beta + 1) * P * R / (beta*beta * P + R)

    return NMI, F


def evaluate_recall(X,T,neighbours):
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
    return recall


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


def evaluate_cos_Inshop(model, query_dataloader, gallery_dataloader):
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
                
    return recall
