# classification.py

import math

from torch import nn
import numpy as np
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import pdb

__all__ = ['Classification', 'BinaryClassify', 'Softmax', 'CrossEntropy', 'AM_Softmax_marginatt', \
    'SphereFace', 'CosFace', 'ArcFace', \
    ]

class Classification(nn.Module):
    def __init__(self, if_cuda=False):
        super(Classification, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, inputs, targets):
        targets = targets.long()
        loss = self.loss(inputs, targets)
        return loss

class BinaryClassify(nn.Module):
    def __init__(self, weight_file=None, if_cuda=False):
        super(BinaryClassify, self).__init__()
        loss_weight = torch.Tensor(np.load(weight_file))
        self.loss = nn.BCELoss(weight=loss_weight)

    def __call__(self, inputs, targets):
        loss = self.loss(inputs, targets)
        return loss

class Softmax(nn.Module):
    def __init__(self, nfeatures, nclasses, if_cuda=False):
        super(Softmax, self).__init__()
        self.fc = nn.Linear(nfeatures, nclasses)
        torch.nn.init.xavier_normal_(self.fc.weight)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        y = self.fc(inputs)
        loss = self.loss(y, targets)
        return loss

class CrossEntropy(nn.Module):
    def __init__(self, if_cuda=False):
        super(CrossEntropy, self).__init__()
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        y = self.logsoftmax(inputs)
        loss = -1*torch.mul(targets, y)
        return loss

class AM_Softmax_old(nn.Module):
    def __init__(self, nfeatures, nclasses, m=1.0, if_cuda=False):
        super(AM_Softmax_old, self).__init__()
        self.nclasses = nclasses
        self.nfeatures = nfeatures
        self.m = m
        self.if_cuda = if_cuda

        self.weights = nn.parameter.Parameter(torch.Tensor(nclasses, nfeatures))
        torch.nn.init.xavier_normal_(self.weights.data)

        self.scale = nn.parameter.Parameter(torch.Tensor(1))
        torch.nn.init.constant_(self.scale.data, 1.00)

        # self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.nfeatures)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs, labels):
        batch_size = labels.size(0)
        inputs = F.normalize(inputs,p=2,dim=1)
        weights_ = F.normalize(self.weights, p=2,dim=1)
        dist_mat = torch.mm(inputs, weights_.transpose(1,0))

        # create one_hot class label
        label_one_hot = torch.FloatTensor(batch_size, self.nclasses).zero_()
        if self.if_cuda:
            label_one_hot = label_one_hot.cuda()
        labels = labels.long()
        if len(labels.size()) == 1:
            labels = labels[:,None]
        label_one_hot.scatter_(1,labels,1) # Tensor.scatter_(dim,index,src)
        label_one_hot = Variable(label_one_hot)


        logits_pos = dist_mat[label_one_hot==1]
        logits_neg = dist_mat[label_one_hot==0]
        if self.if_cuda:
            scale_ = torch.log(torch.exp(Variable(torch.FloatTensor([1.0]).cuda()))
                + torch.exp(self.scale))
        else:
            scale_ = torch.log(torch.exp(Variable(torch.FloatTensor([1.0])))
                + torch.exp(self.scale))

        logits_pos = logits_pos.view(batch_size, -1)
        logits_neg = logits_neg.view(batch_size, -1)
        logits_pos = torch.mul(logits_pos, scale_)
        logits_neg = torch.mul(logits_neg, scale_)
        logits_neg = torch.log(torch.sum(torch.exp(logits_neg), dim=1))[:,None]

        loss = torch.mean(F.softplus(torch.add(logits_neg - logits_pos, self.m)))+1e-2*scale_*scale_
        
        return loss, scale_

class AM_Softmax_marginatt(nn.Module): 
    """Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """

    def __init__(self, nfeatures, nclasses, s=64.0, lambda_regular=100, ndemog=1):
        super(AM_Softmax_marginatt, self).__init__()
        self.s = s
        self.weights = nn.parameter.Parameter(torch.Tensor(nclasses, nfeatures))
        nn.init.xavier_uniform_(self.weights)
        self.lambda_regular = lambda_regular

        self.margin = nn.parameter.Parameter(torch.Tensor(ndemog))
        torch.nn.init.constant_(self.margin.data, math.log(0.35))

        self.loss = nn.CrossEntropyLoss()

    def forward(self, input, label, demog_label):
        label = label.long()
        input = F.normalize(input,p=2,dim=1)
        weights = F.normalize(self.weights,p=2,dim=1)
        cosine = torch.mm(input, weights.t())
        
        # --------------------------- convert label to one-hot ---------------------------
        # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        
        temp = torch.exp(self.margin)
        output = cosine
        demogs = list(set(demog_label.tolist()))
        for demog in demogs:
            indices = (demog_label==demog).nonzero().squeeze()
            output[indices,:] = self.s * (cosine[indices,:] - one_hot[indices,:] * temp[demog])

        loss = self.loss(output, label)

        regularizer = -1*self.lambda_regular * torch.mean(temp)

        loss += regularizer

        return loss, temp

class SphereFace(nn.Module):
    def __init__(self, gamma=0, if_cuda=False):
        super(AngleLoss, self).__init__()
        self.gamma = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0
        self.if_cuda = if_cuda

    def forward(self, input, target):
        self.it += 1
        cos_theta, phi_theta = input
        target = target.view(-1, 1)  # size=(B,1)

        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = index.byte()
        if self.if_cuda:
            index = index.cuda()
        index = Variable(index)

        self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.1 * self.it))
        output = cos_theta * 1.0  # size=(B,Classnum)
        output[index] -= cos_theta[index] * (1.0 + 0) / (1 + self.lamb)
        output[index] += phi_theta[index] * (1.0 + 0) / (1 + self.lamb)

        logpt = F.log_softmax(output, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1 - pt)**self.gamma * logpt
        loss = loss.mean()

        return loss

class CosFace(nn.Module):
    """Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """

    def __init__(self, nfeatures, nclasses, s=64.0, m=0.35):
        super(AM_Softmax_arcface, self).__init__()
        self.s = s
        self.m = m
        self.weights = nn.parameter.Parameter(torch.Tensor(nclasses, nfeatures))
        nn.init.xavier_uniform_(self.weights)
        #stdv = 1. / math.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, input, label):
        label = label.long()
        input = F.normalize(input,p=2,dim=1)
        weights = F.normalize(self.weights,p=2,dim=1)
        cosine = torch.mm(input, weights.t())
        # cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # --------------------------- convert label to one-hot ---------------------------
        # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = self.s * (cosine - one_hot * self.m)

        loss = self.loss(output, label)

        return loss

def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output

class ArcFace(nn.Module):
    r"""Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            device_id: the ID of GPU where the model will be trained by model parallel. 
                       if device_id=None, it will be trained on CPU without model parallel.
            s: norm of input feature
            m: margin
            cos(theta+m)
        """
    def __init__(self, in_features, out_features, s = 64.0, m = 0.50, easy_margin = False):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.s = s
        self.m = m
        
        self.kernel = nn.parameter.Parameter(torch.FloatTensor(in_features, out_features))
        #nn.init.xavier_uniform_(self.kernel)
        nn.init.normal_(self.kernel, std=0.01)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.loss = nn.CrossEntropyLoss()

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis = 1)
        kernel_norm = l2_norm(self.kernel, axis = 0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m #cos(target+margin)
        if self.easy_margin:
            final_target_logit = torch.where(target_logit > 0, cos_theta_m, target_loit)
        else:
            final_target_logit = torch.where(target_logit > self.th, cos_theta_m, target_logit - self.mm)

        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        
        output = cos_theta * self.s
        original_logits = origin_cos * self.s
        loss = self.loss(output, label)
        return loss