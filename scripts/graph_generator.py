import torch
import torch.nn.functional as F


class GraphGenerator():
    def __init__(self, thresh='no'):
        self.thresh = thresh
    
    def _get_A_global(self, W):
        if self.thresh != 'no':
            self.W_to_A = torch.where(self.W_to_A > self.thresh, self.W_to_A, torch.tensor(0).float())
            A = torch.ones_like(self.W_to_A).where(self.W_to_A > self.thresh, torch.tensor(0).float())
        else:
            A = torch.ones_like(self.W_to_A)
        return W, A

    def _get_W(self, x):
        x = (x - x.mean(dim=1).unsqueeze(1))
        norms = x.norm(dim=1)
        self.W_to_A = torch.mm(x, x.t()) / torch.ger(norms, norms)

        x1 = x.transpose(0, 1).unsqueeze(-1)
        x2 = x.transpose(0, 1).unsqueeze(1)

        W = torch.bmm(x1, x2).permute(1, 2, 0)
        W = W / torch.ger(norms, norms).unsqueeze(-1).repeat(1, 1, W.shape[-1])
        return W
    
    def get_graph(self, x):
        W = self._get_W(x)
        W, A = self._get_A_global(W)
        A = torch.nonzero(A)
        W = W[A[:, 0], A[:, 1]]

        return W, A, x
