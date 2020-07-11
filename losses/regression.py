# regression.py

import torch
from torch import nn
import itertools

__all__ = ['Regression', 'AttMatrixCov', 'AttMatrixNorm']


class Regression(nn.Module):

    def __init__(self):
        super(Regression, self).__init__()
        self.loss = nn.MSELoss()

    def __call__(self, inputs, target):
        loss = self.loss.forward(inputs, target)

        return loss

class AttMatrixCov(nn.Module):

    def __init__(self, if_cuda=False):
        super(AttMatrixCov, self).__init__()
        self.loss = nn.MSELoss()
        self.if_cuda = if_cuda

    def __call__(self, attc_list, atts_list):
        assert len(attc_list) == len(atts_list)
        natt = len(attc_list)
        ntemp = attc_list[0].size(0)
        comp_idx = list(itertools.combinations(list(range(0,ntemp)), 2)) # ?x2

        loss_val = 0.0

        if atts_list[0] is not None:
            assert attc_list[0].size(0) == atts_list[0].size(0)

            for i in range(natt):
                attc = attc_list[i].squeeze()
                atts = atts_list[i].squeeze()
                temp1 = torch.eye(attc.size(1))
                temp2 = torch.eye(atts.size(2))
                if self.if_cuda:
                    temp1 = temp1.cuda()
                    temp2 = temp2.cuda()
                for idx in comp_idx:
                    attc_demog1 = attc[idx[0],:].unsqueeze(0)
                    attc_demog2 = attc[idx[1],:].unsqueeze(0)
                    loss_val += self.loss(torch.matmul(torch.t(attc_demog1), attc_demog2), temp1)

                    atts_demog1 = atts[idx[0],:]
                    atts_demog2 = atts[idx[1],:]
                    loss_val += self.loss(torch.matmul(torch.t(atts_demog1), atts_demog2), temp2)
        else:
            for i in range(natt):
                attc = attc_list[i].squeeze()
                temp1 = torch.eye(attc.size(1))
                if self.if_cuda:
                    temp1 = temp1.cuda()
                for idx in comp_idx:
                    attc_demog1 = attc[idx[0],:].unsqueeze(0)
                    attc_demog2 = attc[idx[1],:].unsqueeze(0)
                    loss_val += self.loss(torch.matmul(torch.t(attc_demog1), attc_demog2), temp1)
        return loss_val

class AttMatrixNorm(nn.Module):

    def __init__(self, if_cuda=False):
        super(AttMatrixNorm, self).__init__()
        self.loss = nn.MSELoss()
        self.if_cuda = if_cuda

    def __call__(self, attc_list, atts_list):
        assert len(attc_list) == len(atts_list)
        natt = len(attc_list)
        ntemp = attc_list[0].size(0)
        comp_idx = list(itertools.combinations(list(range(0,ntemp)), 2)) # ?x2

        loss_val = 0.0

        if atts_list[0] is not None:
            assert attc_list[0].size(0) == atts_list[0].size(0)

            for i in range(natt):
                attc = attc_list[i].squeeze()
                atts = atts_list[i].squeeze()
                
                temp = torch.eye(atts.size(2))
                if self.if_cuda:
                    temp = temp.cuda()
                    
                for idx in comp_idx:
                    attc_demog1 = attc[idx[0],:]
                    attc_demog2 = attc[idx[1],:]
                    loss_val += -1*self.loss(attc_demog1, attc_demog2)

                    atts_demog1 = atts[idx[0],:]
                    atts_demog2 = atts[idx[1],:]
                    loss_val += self.loss(torch.matmul(torch.t(atts_demog1), atts_demog2), temp)
        else:
            for i in range(natt):
                attc = attc_list[i].squeeze()
                
                for idx in comp_idx:
                    attc_demog1 = attc[idx[0],:]
                    attc_demog2 = attc[idx[1],:]
                    loss_val += -1*self.loss(attc_demog1, attc_demog2)
        return loss_val
