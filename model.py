# model.py

import math

import evaluate

import losses
import models
from torch import nn
import torch.optim as optim

from itertools import combinations
import torch

import pdb

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 1)
        m.bias.data.zero_()


class Model:
    def __init__(self, args):
        self.args = args

        self.ngpu = args.ngpu
        self.cuda = args.cuda
        self.model_type = args.model_type
        self.model_options = args.model_options
        self.loss_type = args.loss_type
        self.loss_options = args.loss_options
        self.evaluation_type = args.evaluation_type
        self.evaluation_options = args.evaluation_options

    def setup(self, checkpoints):
        model = getattr(models, self.model_type)(**self.model_options)

        init_kernels(self.args, model)
        
        criterion = {}
        keys_loss = list(self.loss_type)
        for key in keys_loss:
            criterion[key] = getattr(losses, self.loss_type[key])(**self.loss_options[key])
        
        evaluation = getattr(evaluate, self.evaluation_type)(**self.evaluation_options)

        if self.cuda:
            model = nn.DataParallel(model, device_ids=list(range(self.ngpu)))
            model = model.cuda()
            for key in keys_loss:
                criterion[key] = criterion[key].cuda()

        model_dict = {}
        model_dict['model'] = model
        model_dict['loss'] = criterion

        # remove fuse_mark from the optimizer
        new_params = []
        params = list(model.named_parameters())
        for p in params:
            if p[0].endswith('fuse_mark'):
                continue
            new_params.append(p[1])
        for key in keys_loss:
            params = list(criterion[key].parameters())
            new_params.extend(params)

        lr = self.args.learning_rate
        model_dict['optimizer'] = getattr(optim, self.args.optim_method)(
            new_params, lr=lr, **self.args.optim_options)
        
        if checkpoints.latest('resume') is None:
            pass
        else:
            model_dict = checkpoints.load(model_dict, checkpoints.latest('resume'))

        return model_dict, evaluation

def init_kernels(args, model):
    state_dict = model.state_dict()
    keys = list(state_dict)
    keys_mask = [x for x in keys if x.endswith('kernel_mask')]
    fuse_keys = [x for x in keys if x.endswith('fuse_mark')]
    for i,key_mask in enumerate(keys_mask):
        kernels = state_dict[key_mask]
        kernels = torch.mean(kernels, dim=0).unsqueeze(0)
        kernels = kernels.repeat(args.ndemog,1,1,1)
        state_dict[key_mask] = kernels
    model.load_state_dict(state_dict)