import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from scripts.utils import l2_norm


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T


class DiverseLoss(nn.Module):
    def __init__(self, embedding_size=512, num_per_class=2):
        super(DiverseLoss, self).__init__()
        self.embedding_size = embedding_size
        self.num_per_class = num_per_class

    def forward(self, hs, bs, edge_index):
        r, c = edge_index[:, 0], edge_index[:, 1]
        mean = torch.reshape(torch.tile(scatter_mean(hs, r, dim=0), [1, bs]), [-1, self.embedding_size])
        loss_std = 1 - torch.sqrt((torch.pow((hs - mean), 2)).sum() / hs.shape[0])
        return loss_std


def cross_entropy(logits, label, size_average=True):
    if size_average:
        return torch.mean(torch.sum(-label * F.log_softmax(logits, -1), -1))
    else:
        return torch.sum(torch.sum(-label * F.log_softmax(logits, -1), -1))


class NpairLoss(nn.Module):
    def __init__(self, l2_reg=3e-3, num_per_class=2):
        super(NpairLoss, self).__init__()
        self.l2_reg = l2_reg
        self.n_per_class = num_per_class
        self.margin = 1
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x, label, equal_shape=True, bs=None, embedding_z=None):
        labels = torch.chunk(label, self.n_per_class, dim=0)  
        if equal_shape:
            samples = torch.chunk(x, self.n_per_class, dim=0)
            anchor = samples[0]
            batch_size = samples[0].size(0)
            l2_loss = 0
            for id in range(self.n_per_class):
                l2_loss += torch.sum(samples[id]**2) / batch_size
            label = labels[0].view(labels[0].size(0), 1)
            label = (label == torch.transpose(label, 0, 1)).float()
            label = label / torch.sum(label, dim=1, keepdim=True).float()
            loss_ce = 0
            for id in range(1, self.n_per_class):      
                logit = torch.matmul(anchor, torch.transpose(samples[id], 0, 1))
                loss_ce += cross_entropy(logit, label)      
            loss = loss_ce / (self.n_per_class - 1) + self.l2_reg * l2_loss * 0.25
            return loss
        else:
            loss = 0
            anchor_list = torch.chunk(embedding_z, self.n_per_class, dim=0)
            neg_tile_list = torch.chunk(x, self.n_per_class, dim=0)
            for j in range(self.n_per_class):
                if j == self.n_per_class-1:
                    id_p = 0
                else:
                    id_p = j + 1
                anchor = anchor_list[j]
                positive = anchor_list[id_p]
                neg_tile = neg_tile_list[j]
                label = labels[0]
                sampled_class_num = anchor.shape[0]

                positive_tile = torch.reshape(torch.tile(positive, [1, sampled_class_num]), (-1, embedding_z.shape[-1]))
                diag = torch.zeros(sampled_class_num**2, device="cuda")
                index = torch.tensor([i*len(label)+i for i in range(len(label))], device="cuda")
                diag[index] = 1
                diag = diag[:,None].repeat(1, positive_tile.shape[-1])
                neg_tile = torch.where(diag==1, positive_tile, neg_tile)
                l2_loss = torch.sum(anchor**2) / anchor.shape[0] + torch.sum(neg_tile**2) / neg_tile.shape[0]

                out = torch.matmul(anchor.unsqueeze(1), neg_tile.reshape(sampled_class_num, sampled_class_num, neg_tile.shape[-1]).permute(0,2,1)).squeeze()
                label_ce = torch.range(0, sampled_class_num-1, device="cuda").long()
                ce_loss = self.ce(out, label_ce)
            
                loss += ce_loss + self.l2_reg * l2_loss * 0.25
            loss = loss / self.n_per_class
            return loss


class Proxy_Anchor(nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32, num_per_class = 2):
        torch.nn.Module.__init__(self)
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        self.num_per_class = num_per_class
        
    def forward(self, X, label, P):
        
        T = label
        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        N_one_hot = 1 - P_one_hot

        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term   
        return loss
    