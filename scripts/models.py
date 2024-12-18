import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

from scripts.backbone import googlenet, Resnet50
from scripts import loss
from scripts.utils import *
from scripts.graph_generator import GraphGenerator
import warnings
warnings.filterwarnings("ignore")


class Feature_Extractor(nn.Module):
    def __init__(self, embedding_size=128, n_mid=1024, pretrained=False, model='googlenet', is_norm=False, is_pooling=True, bn_freeze=True):
        super(Feature_Extractor, self).__init__()
        self.is_norm = is_norm
        self.is_pooling = is_pooling
        self.fc = nn.Sequential(
            nn.BatchNorm1d(n_mid),
            nn.Linear(n_mid, embedding_size))
        if model == 'Googlenet':
            self.backbone = googlenet(pretrained=pretrained)
        elif model == 'Resnet50':
            self.backbone = Resnet50(pretrained=pretrained, Pool=is_pooling, bn_freeze=bn_freeze)
        elif model == "DINO":
            self.backbone = torch.hub.load("facebookresearch/dino:main", "dino_vits16", pretrained=pretrained)
            self.rm_head(self.backbone)
            nn.init.constant_(self.fc[1].bias.data, 0)
            nn.init.orthogonal_(self.fc[1].weight.data)
        else:
            raise NotImplementedError
        
    def forward(self, x):
        embedding_y_orig = self.backbone(x)
        embedding_z = self.fc(embedding_y_orig)
        if self.is_norm:
            embedding_z = l2_norm(embedding_z)
        return embedding_z  
    
    def rm_head(self, m):
        names = set(x[0] for x in m.named_children())
        target = {"head", "fc", "head_dist"}
        for x in names & target:
            m.add_module(x, nn.Identity())


class GeneralPulling(nn.Module):
    def __init__(self, embedding_size=128, alpha=90.0, num_per_class=2):
        super(GeneralPulling, self).__init__()
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.num_per_class = num_per_class

    def forward(self, x, jm, edge_params):
        samples = torch.chunk(x, self.num_per_class, dim=0)
        edge_params_split = torch.chunk(edge_params, self.num_per_class, dim=0)
        anchor = samples[0]
        sample_class_num = anchor.shape[0]
        bs = sample_class_num*self.num_per_class
        
        bool_masks = []
        neg_hat_list = []
        all_neg_list = []
        for id in range(0, self.num_per_class):
            if id == self.num_per_class-1:
                id_p = 0
            else:
                id_p = id + 1
            positive = samples[id_p]
            anchor = samples[id]
            anc_tile = torch.reshape(torch.tile(anchor, [1, sample_class_num]), [-1, self.embedding_size])
            pos_tile = torch.reshape(torch.tile(positive, [1, sample_class_num]), [-1, self.embedding_size])
            dist_pos = F.pairwise_distance(anc_tile, pos_tile, 2)
            edge_params = edge_params_split[id].reshape(sample_class_num, bs, -1)

            for neg_id in range(self.num_per_class):
                negative = samples[neg_id]
                neg_tile = torch.tile(negative, [sample_class_num, 1])
                dist_neg = F.pairwise_distance(anc_tile, neg_tile, 2)

                edge_params_group = edge_params[:,sample_class_num*neg_id:sample_class_num*(neg_id+1),:].reshape(-1, self.embedding_size)
                para_mode = edge_params_group * np.exp(-self.alpha / jm)

                r = ((dist_pos.unsqueeze(-1).repeat(1, self.embedding_size) \
                        + (dist_neg - dist_pos).unsqueeze(-1).repeat(1, self.embedding_size) \
                        * para_mode) / dist_neg.unsqueeze(-1).repeat(1, self.embedding_size)) 
                
                dis_vector = torch.mul((neg_tile - anc_tile), r)
                neg2_tile = anc_tile + dis_vector
                neg_mask = torch.ge(dist_pos, dist_neg)
                bool_masks.append(neg_mask)
                op_neg_mask = ~ neg_mask
                neg_mask = neg_mask.float().unsqueeze(-1).repeat(1, self.embedding_size)
                op_neg_mask = op_neg_mask.float().unsqueeze(-1).repeat(1, self.embedding_size)
                neg_hat = torch.mul(neg_tile, neg_mask) + torch.mul(neg2_tile, op_neg_mask)
                all_neg_list.append(neg_hat)
                if neg_id == 0:
                    neg_tmp = neg_hat
                else:
                    neg_hat += np.random.random() * (neg_tmp - neg_hat)
                    neg_tmp = neg_hat
            neg_hat_list.append(neg_tmp)
        return torch.cat(neg_hat_list, dim=0), torch.cat(all_neg_list, dim=0)


class MetaLayer(nn.Module):
    def __init__(self, edge_model=None, node_model=None):
        super(MetaLayer, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, feats, edge_index, edge_attr):
        feats, edge_index, edge_attr = self.node_model(feats, edge_index, edge_attr)
        _, edge_attr = self.edge_model(feats, edge_index, edge_attr)
        return feats, edge_index, edge_attr


class Vanilla_Attention(nn.Module):
    def __init__(self, embed_dim, nhead, dropout=0.1):
        super(Vanilla_Attention, self).__init__()
        self.embed_dim = embed_dim
        self.hdim = embed_dim // nhead
        self.nhead = nhead

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.functional.softmax
        self.reset_parameters()
    
    def forward(self, queries, keys, values):
        q, k, v = self.q_linear(queries), self.k_linear(keys), self.v_linear(values)
        d = q.shape[-1]

        scores = torch.bmm(q, k.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = self.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(self.attention_weights), v).squeeze(1)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.constant_(self.q_linear.bias, 0.)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.constant_(self.k_linear.bias, 0.)
        nn.init.xavier_uniform_(self.v_linear.weight)
        nn.init.constant_(self.v_linear.bias, 0.)


class MultiHeadDotProduct(nn.Module):
    def __init__(self, embed_dim, nhead, aggr, mult_attr=0, cfg=None):
        super(MultiHeadDotProduct, self).__init__()
        self.embed_dim = embed_dim
        self.hdim = embed_dim // nhead
        self.nhead = nhead
        self.nb_classes = cfg.TRAIN.NUM_SAMPLED_CLASSES
        self.sample_per_class = cfg.TRAIN.NUM_PER_CLASS
        self.aggr = aggr
        self.mult_attr = mult_attr

        # FC Layers for input
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)

        # fc layer for concatenated output
        self.out = nn.Linear(embed_dim, embed_dim)

        self.reset_parameters()

    def forward(self, feats: torch.tensor, edge_index: torch.tensor,
            edge_attr: torch.tensor):
        q = k = v = feats
        bs = q.size(0)

        k = self.k_linear(k).view(bs, self.nhead, self.hdim).transpose(0, 1)
        q = self.q_linear(q).view(bs, self.nhead, self.hdim).transpose(0, 1)
        v = self.v_linear(v).view(bs, self.nhead, self.hdim).transpose(0, 1)
        
        # perform multi-head attention
        feats = self._attention(q, k, v, edge_index, edge_attr, bs)
        feats = feats.transpose(0, 1).contiguous().view(
            bs, self.nhead * self.hdim)
        feats = self.out(feats)

        return feats

    def _attention(self, q, k, v, edge_index=None, edge_attr=None, bs=None):
        r, c, e = edge_index[:, 0], edge_index[:, 1], edge_index.shape[0]

        scores = torch.matmul(
            q.index_select(1, c).unsqueeze(dim=-2),
            k.index_select(1, r).unsqueeze(dim=-1))
        scores = scores.view(self.nhead, e, 1) / math.sqrt(self.hdim)
        scores = softmax(scores, c, 1, bs, self.nb_classes, self.sample_per_class)
        
        if self.mult_attr:
            scores = scores * edge_attr.unsqueeze(1)

        out = scores * v.index_select(1, r)
        out = self.aggr(out, c, 1, bs)
        if type(out) == tuple:
            out = out[0]
        return out

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.constant_(self.q_linear.bias, 0.)
        nn.init.xavier_uniform_(self.v_linear.weight)
        nn.init.constant_(self.v_linear.bias, 0.)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.constant_(self.k_linear.bias, 0.)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.constant_(self.out.bias, 0.)


class Messagepasstoedge(nn.Module):
    def __init__(self, embed_dim=2048, act=F.relu, num_heads=4, d_hid=None):
        super(Messagepasstoedge, self).__init__()
        self.att = Vanilla_Attention(embed_dim, num_heads)
        d_hid = embed_dim * 4 if d_hid is None else d_hid
        
        self.act = act
        self.node_to_edge = nn.Linear(embed_dim, embed_dim)
        self.norm = Sequential(
            nn.GELU(), 
            nn.LayerNorm(embed_dim))
        self.linear = Sequential(
            nn.Linear(embed_dim, d_hid), 
            nn.GELU(), 
            nn.LayerNorm(d_hid), 
            nn.Linear(d_hid, embed_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        self.node_to_edge.reset_parameters()
        for item in self.linear:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, feats, edge_index, edge_attr):
        r, c = edge_index[:, 0], edge_index[:, 1]
        
        edge_feats = self.act(self.node_to_edge(feats))
        edge_feats1 = edge_feats[r].unsqueeze(1)
        edge_feats2 = edge_feats[c].unsqueeze(1)

        input_feats = torch.cat([edge_feats1, edge_feats2], dim=1)
        edge_feats = self.att(edge_attr.unsqueeze(1), input_feats, input_feats)
        edge_feats_att = self.norm(edge_feats + edge_attr)
        edge_attr = self.norm(self.linear(edge_feats_att) + edge_feats_att)
        return feats, edge_attr


class GNNDecoder(nn.Module):
    def __init__(self, cfg=None, embed_dim=2048, out_dim=512, reduce=4, num_layers=2, num_class=98):
        super(GNNDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.gg = GraphGenerator()

        self.dim_red = nn.Linear(embed_dim, int(embed_dim / reduce))
        self.embed_dim = int(self.embed_dim / reduce)

        # aggregation
        self.aggr = lambda out, row, dim, x_size: scatter_add(out, row, dim=dim, dim_size=x_size)
        self.node_models = [GATNwtwork(self.embed_dim, self.aggr, cfg=cfg) for _ in range(num_layers)]
        self.edge_models = [Messagepasstoedge(self.embed_dim, F.gelu) for _ in range(num_layers)]
        self.gnn = Sequential(*[MetaLayer(node_model=self.node_models[i], edge_model=self.edge_models[i]) \
                     for i in range(num_layers)])

        dim = self.embed_dim
        self.bottleneck1 = Sequential(*[nn.BatchNorm1d(dim)])
        for layer in self.bottleneck1:
            layer.bias.requires_grad_(False)
            layer.apply(weights_init_kaiming)
        self.bottleneck2 = Sequential(*[nn.BatchNorm1d(self.embed_dim)])
        for layer in self.bottleneck2:
            layer.bias.requires_grad_(False)
            layer.apply(weights_init_kaiming)
        self.fc = Sequential(*[nn.Linear(dim, out_dim)])
        self.cls = Sequential(*[nn.Linear(self.embed_dim, num_class)])
        self.out_para = Sequential(*[nn.Sigmoid()])

    def forward(self, feats, edge_index, edge_attr=None):
        if self.dim_red is not None:
            feats = self.dim_red(feats)

        for layer in self.gnn:
            feats, edge_index, edge_attr = layer(feats, edge_index, edge_attr)

        edge_features = self.bottleneck1(edge_attr)
        node_features = self.bottleneck2(feats)
        edge_para = self.out_para(self.fc(edge_features))
        node_pred = self.cls(node_features)
        return node_pred, edge_para 


class GATNwtwork(nn.Module):
    def __init__(self, embed_dim, aggr, cfg=None):
        super(GATNwtwork, self).__init__()
        layers = [DotAttentionLayer(embed_dim, aggr, cfg=cfg)]
        self.layers = Sequential(*layers)
    
    def forward(self, feats, edge_index, edge_attr=None):
        for layer in self.layers:
            feats, edge_index, edge_attr = layer(feats, edge_index, edge_attr)
        return feats, edge_index, edge_attr

        
class DotAttentionLayer(nn.Module):
    def __init__(self, embed_dim, aggr, cfg=None, d_hid=None):
        super(DotAttentionLayer, self).__init__()
        num_heads = cfg.MODEL.ATT_HEAD
        self.att = MultiHeadDotProduct(embed_dim, num_heads, aggr, cfg=cfg)
        d_hid = 4 * embed_dim if d_hid is None else d_hid

        self.linear1 = nn.Linear(embed_dim, d_hid)
        self.linear2 = nn.Linear(d_hid, embed_dim)

        self.act = nn.GELU()
        self.edge_to_node = nn.Linear(embed_dim, embed_dim)
        self.update_node = nn.Linear(embed_dim, embed_dim)
        self.norm = Sequential(
            nn.GELU(), 
            nn.LayerNorm(embed_dim))
        self.reset_parameters()
   
    def reset_parameters(self):
        self.edge_to_node.reset_parameters()
        self.update_node.reset_parameters()
    
    def forward(self, feats, egde_index, edge_attr):
        feats_ = self.att(feats, egde_index, edge_attr)
        feats_ = self.norm(feats + feats_)

        # edge message pass to node in every layer
        node_attr = self.act(self.edge_to_node(edge_attr))
        feats_att = self.norm(feats_ + scatter_add(node_attr, egde_index[:, 0], dim=0)) 

        feats_ = self.linear2(self.act(self.linear1(feats_att)))
        feats = self.norm(feats_att + feats_)
        return feats, egde_index, edge_attr


class GCA_HNG(nn.Module):
    def __init__(self, cfg, embedding_size=120, n_class=99,
                 beta=1e+4, softmax_factor=1e+4,
                 pretrained=False, num_per_class=3, gnn_layers=2, is_norm=False, is_pooling=True, bn_freeze=True):
        super(GCA_HNG, self).__init__()
        self.is_pooling = is_pooling
        self.embedding_size = embedding_size
        backbone = cfg.MODEL.BACKBONE
        if backbone == 'Googlenet':
            n_mid = 1024
            self.feature_extractor = Feature_Extractor(self.embedding_size, n_mid, pretrained, model=backbone, is_norm=is_norm)
        elif backbone == 'Resnet50':
            n_mid = 2048
            self.feature_extractor = Feature_Extractor(self.embedding_size, n_mid, pretrained, model=backbone, is_norm=is_norm, is_pooling=is_pooling, bn_freeze=bn_freeze)
        elif backbone == 'DINO':
            n_mid = 384
            self.feature_extractor = Feature_Extractor(self.embedding_size, n_mid, pretrained, model=backbone, is_norm=is_norm)
        else:
            raise NotImplementedError
        
        self.beta = beta
        self.num_per_class = num_per_class
        self.softmax_factor = softmax_factor
        self.cfg = cfg
        self.n_class = n_class
        self.clip = self.cfg.SOLVER.GRAD_CLIP
        self.loss_fn = loss.NpairLoss(num_per_class=num_per_class)
        
        self.ce1 = nn.CrossEntropyLoss()
        self.softmax_classifier = nn.Linear(embedding_size, n_class)

        self.gg = GraphGenerator()
        self.gnn = GNNDecoder(cfg=cfg, embed_dim=embedding_size, out_dim=embedding_size, reduce=1, num_layers=gnn_layers, num_class=n_class)
        self.pulling = GeneralPulling(embedding_size=embedding_size, num_per_class=num_per_class, alpha=cfg.LOSS.ALPHA)
        
        self.std_loss = loss.DiverseLoss(embedding_size, num_per_class)
        self.ce2 = nn.CrossEntropyLoss()
        self.mining_loss = loss.NpairLoss(num_per_class=num_per_class)

    def forward(self, x, t, j_avg=None, j_gen=None, opt_c=None, opt_s=None, opt_gnn=None):
        metrics = {}
        grad_clip = self.cfg.SOLVER.GRAD_L2_CLIP
        label = t.squeeze(-1)
        if opt_c is not None:
            opt_c.zero_grad()
        if opt_gnn is not None:
            opt_gnn.zero_grad()

        embedding_z = self.feature_extractor(x)
        bs = embedding_z.shape[0]
        sample_class_num = bs // self.num_per_class

        jm = self.loss_fn(embedding_z, label) 
        if j_gen is not None:
            e_bj = np.exp(-self.beta / j_gen)
        metrics['J_m'] = jm.item()

        edge_attr, edge_index, node_emb = self.gg.get_graph(embedding_z)
        node_pred, edge_para = self.gnn(node_emb, edge_index, edge_attr)
        J_gce = self.ce2(node_pred, label)
        metrics['J_gnnce'] = J_gce.item()

        embedding_z_quta, _ = self.pulling(embedding_z, j_avg, edge_para.detach())
        jsyn = (1.0 - e_bj) * self.mining_loss(embedding_z_quta, t, False, bs, embedding_z=embedding_z)
        metrics['J_syn'] = jsyn.item()
        jmetric = jm + jsyn + J_gce

        metrics['J_metric'] = jmetric.item()
        if opt_c is not None:
            jmetric.backward(retain_graph=True)
            if self.clip:
                torch.nn.utils.clip_grad_value_(self.feature_extractor.parameters(), grad_clip) 
            opt_c.step()
        if opt_gnn is not None:
            if self.clip:
                torch.nn.utils.clip_grad_value_(self.gnn.parameters(), grad_clip)
            opt_gnn.step()
            
        # train soft cls
        if opt_s is not None:
            opt_s.zero_grad()
        logits_orig = self.softmax_classifier(embedding_z.detach())
        ce = 10 * self.ce1(logits_orig, label)
        metrics['J_softce'] = ce.item()
        if opt_s is not None:
            ce.backward(retain_graph=True)
            opt_s.step()

            if opt_gnn is not None:
                opt_gnn.zero_grad()

        edge_attr, edge_index, node_emb = self.gg.get_graph(embedding_z.detach())
        node_pred, edge_para = self.gnn(node_emb, edge_index, edge_attr)
        embedding_z_quta, all_hard_neg = self.pulling(embedding_z.detach(), j_avg, edge_para)

        J_div = 1e-2 * self.std_loss(edge_para, bs, edge_index)
        metrics['J_div'] = J_div.item()

        logits_mix = self.softmax_classifier(embedding_z_quta)
        label_mix = torch.cat([label[0:sample_class_num]]*bs)
        jsoft = self.softmax_factor * self.ce2(logits_mix, label_mix)
        metrics['J_gence'] = jsoft.item()

        temp_list = []
        for k in range(self.num_per_class):
            temp = torch.reshape(torch.tile(embedding_z[k*sample_class_num:(k+1)*sample_class_num].detach(), [1, sample_class_num]), [-1, self.embedding_size])
            temp_list.append(temp)
        embedding_z_tile = torch.cat(temp_list, dim=0)
        J_sim_mix = 1 - torch.cosine_similarity(l2_norm(embedding_z_tile), l2_norm(embedding_z_quta), dim=1)
        diag_index = torch.nonzero(torch.tile(torch.eye(sample_class_num, device="cuda"), [1, self.num_per_class]))
        mask_index = diag_index[:,0]*bs+diag_index[:,1]
        neg_mask = torch.ones_like(J_sim_mix, device="cuda")
        neg_mask[mask_index] = 0
        neg = torch.zeros_like(J_sim_mix, device="cuda")
        J_sim_mix = torch.where(neg_mask==1, J_sim_mix, neg)

        jgen = (jsoft + self.cfg.LOSS.DOT_WEIGHT * J_sim_mix.mean() + self.cfg.LOSS.DIV_WEIGHT * J_div) * 0.5
        metrics['J_gen'] = jgen.item()
        if opt_gnn is not None:
            jgen.backward()
            if self.clip:
                torch.nn.utils.clip_grad_value_(self.gnn.parameters(), grad_clip) 
            opt_gnn.step()
        return jgen, jmetric, jm, ce, e_bj, metrics

