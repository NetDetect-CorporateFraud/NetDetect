import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, threshold=0.5,
                 size_average=True, aggregate=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.threshold = threshold
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.aggregate = aggregate

    def forward(self, input, target):
        if input.dim()==2 and input.size(1) == 2: # shape: (N, 2)
            pass
        elif input.dim()==1 or input.size(1) == 1: # shape: (N) or (N,1)
            input_y_1 = input.view(-1,1) # pt = p if y=1
            input_y_0 = (2*self.threshold - input).view(-1,1) # pt = 1-p if y=0
            # input: (N, 2)
            input = torch.cat([input_y_0, input_y_1], dim=1)

        target = target.long().view(-1, 1)
        logpt = F.log_softmax(input,dim=-1)
        logpt = logpt.gather(dim=1, index=target)
        # log(pt), pt = p if y=1; pt = 1-p, if y=0
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if not self.aggregate:
            return loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class ReweightLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=False):
        super(ReweightLoss, self).__init__()
        # self.T_ = nn.Parameter(data=torch.tensor([[0.95,0.05],[0.05,0.95]]), requires_grad=False)
        self.size_average = size_average
        self.fl = FocalLoss(gamma=gamma, alpha=alpha, aggregate=False)


    def forward(self, out, target, T_=None):
        self.T_ = T_
        out_softmax = F.softmax(out, dim=1)


        pro1 = torch.gather(out_softmax, dim=-1, index=target.unsqueeze(1)).squeeze()

        # [2, 2] * [batch, 2, 1] -> [batch, 2, 1]
        noisy_prob = torch.matmul(self.T_.t(), out_softmax.unsqueeze(-1)).squeeze() if self.T_ is not None else out_softmax
        pro2 = torch.gather(noisy_prob, dim=-1, index=target.unsqueeze(1)).squeeze()


        beta = pro1 / pro2
        beta = Variable(beta, requires_grad=True)


        cross_loss = self.fl(out, target)

        _loss = beta * cross_loss

        if self.size_average:
            return _loss.mean()
        else:
            return _loss.sum()

class ReweightRevisionLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=False):
        super(ReweightRevisionLoss, self).__init__()

        self.fl = FocalLoss(gamma=gamma, alpha=alpha, aggregate=False)
        self.size_average = size_average

    def forward(self, out, target, T_=None, correction=None):

        self.T_ = T_
        out_softmax = F.softmax(out, dim=1)

        self.T_ = self.T_ + correction if correction is not None else self.T

        pro1 = torch.gather(out_softmax, dim=-1, index=target.unsqueeze(1)).squeeze()

        noisy_prob = torch.matmul(self.T_.t(), out_softmax.unsqueeze(-1)).squeeze() if self.T_ is not None else out_softmax
        pro2 = torch.gather(noisy_prob, dim=-1, index=target.unsqueeze(1)).squeeze()
        beta = pro1 / pro2

        beta = Variable(beta, requires_grad=True)

        cross_loss = self.fl(out, target)
        _loss = beta * cross_loss

        if self.size_average:
            return _loss.mean()
        else:
            return _loss.sum()
