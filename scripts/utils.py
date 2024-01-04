import logging
import torch
from torch import nn
from torch_scatter import scatter_max, scatter_add

logger = logging.getLogger('GNNReID.Util')


class Sequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def softmax(src, index, dim, dim_size, nb_classes, sample_per_class, margin: float = 0.):
    diag_index = torch.nonzero(torch.tile(torch.eye(nb_classes, device="cuda"), [sample_per_class, sample_per_class]))
    mask_index = diag_index[:,0]*dim_size+diag_index[:,1]
    neg_mask = torch.ones_like(src, device="cuda")
    neg_mask[:,mask_index,:] = 0
    neg_inf = torch.ones_like(src, device="cuda") - torch.inf
    src = torch.where(neg_mask==1, src, neg_inf)
    src_max = torch.clamp(scatter_max(src.float(), index, dim=dim, dim_size=dim_size)[0], min=0.)
    src = (src - src_max.index_select(dim=dim, index=index)).exp() # index_select for expend dim of src_max
    denom = scatter_add(src, index, dim=dim, dim_size=dim_size)
    out = src / (denom + (margin - src_max).exp()).index_select(dim, index)

    return out


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output
