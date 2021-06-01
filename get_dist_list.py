import os
import sys
import traceback
import random
import config
import utils
from model import Model
from dataloader import Dataloader
from checkpoints import Checkpoints
import torch

from itertools import combinations

import numpy as np
import pdb

args, config_file = config.parse_args()
# Data Loading    
if args.train == 'face_cls':
    from test_cls import Tester
    from train_cls import Trainer

if args.train == 'face_margin':
    from test_margin import Tester
    from train_margin import Trainer

if args.dataset_train == 'ClassSamplesDataLoader':
    from train_classload import Trainer


def main():
    # parse the arguments
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if args.save_results:
        utils.saveargs(args, config_file)

    # initialize the checkpoint class
    checkpoints = Checkpoints(args)

    # Create Model
    models = Model(args)
    model, model_dict, evaluation = models.setup(checkpoints)

    print('Model:\n\t{model}\nTotal params:\n\t{npar:.2f}M'.format(
          model=args.model_type,
          npar=sum(p.numel() for p in model.parameters()) / 1000000.0))

    #### get kernel information ####
    ndemog = args.ndemog
    ndemog = list(range(ndemog))
    demog_combs = list(combinations(ndemog, 2))
    #### get kernel information ####

    dist_list = get_att_dist(model)
    pdb.set_trace()
    

def get_att_dist(model, demog_combs):
    state_dict = model.state_dict()
    keys = list(state_dict)
    adv_keys = [x for x in keys if x.endswith('att_channel')]

    dist_list = []
    for key_mask in adv_keys:
        kernels = state_dict[key_mask]
        kernels = kernels.squeeze()
        dist = 0
        for demog_comb in demog_combs:
            k1 = kernels[demog_comb[0],:].view(1,-1)
            k2 = kernels[demog_comb[1],:].view(1,-1)
            k1 = k1/torch.norm(k1,dim=1)
            k2 = k2/torch.norm(k2,dim=1)
            # dist += torch.norm(k1-k2)
            dist += -1*torch.matmul(k1, torch.transpose(k2,0,1))
        dist = dist/float(len(demog_combs))
        dist_list.append(dist.item())
    return dist_list