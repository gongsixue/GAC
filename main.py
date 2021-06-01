# main.py

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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from itertools import combinations
from ptflops import get_model_complexity_info
from thop import profile
import numpy as np

args, config_file = config.parse_args()    
from test_cls import Tester
from train_cls import Trainer

if args.dataset_train == 'ClassSamplesDataLoader':
    from train_classload import Trainer

import pdb

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
    model_dict, evaluation = models.setup(checkpoints)

    print('Model:\n\t{model}\nTotal params:\n\t{npar:.4f}M'.format(
          model=args.model_type,
          npar=sum(p.numel() for p in model_dict['model'].parameters()) / 1000000.0))

    #### get kernel information ####
    ndemog = args.ndemog
    ndemog = list(range(ndemog))
    demog_combs = list(combinations(ndemog, 2))
    #### get kernel information ####

    #### create writer for tensor boader ####
    if args.save_results:
        writer = SummaryWriter(args.tblog_dir)
    else:
        writer = None
    #### create writer for tensor boader ####

    # The trainer handles the training loop
    trainer = Trainer(args, model_dict['model'], model_dict['loss'], model_dict['optimizer'], writer)
    # The trainer handles the evaluation on validation set
    tester = Tester(args, model_dict['model'], evaluation, writer)

    test_freq = 1

    dataloader = Dataloader(args)

    if args.extract_feat:
        loaders  = dataloader.create(flag='Test')
        tester.extract_features(loaders, 1)
    elif args.just_test:
        loaders  = dataloader.create(flag='Test')
        acc_test,acc_mean = tester.test(loaders, 1)
        print(acc_test, acc_mean)
    else:
        loaders  = dataloader.create()
        if args.dataset_train == 'ClassSamplesDataLoader':
            loaders['train'] = dataloader.dataset_train

        # start training !!!
        acc_best = 0
        loss_best = 999
        stored_models = {}

        for epoch in range(args.nepochs-args.epoch_number):
            epoch += args.epoch_number
            print('\nEpoch %d/%d\n' % (epoch + 1, args.nepochs))

            # train for a single epoch
            loss_train = trainer.train(loaders, epoch)

            acc_test=0
            if float(epoch) % test_freq == 0:
                acc_test,acc_mean = tester.test(loaders, epoch)

            if loss_best > loss_train:
                loss_best = loss_train
                acc_best = acc_test
            if  float(epoch) % test_freq == 0 and args.save_results:
                stored_models['model'] = trainer.model
                stored_models['loss'] = trainer.criterion
                stored_models['optimizer'] = trainer.optimizer
                checkpoints.save(acc_test, stored_models, epoch)

            if epoch == args.fuse_epoch:
                update_kernels(args, trainer.model, demog_combs, ndemog)

    if args.save_results:
        writer.close()

def update_kernels(args, model, demog_combs, ndemog):
    state_dict = model.state_dict()
    keys = list(state_dict)
    fuse_keys = [x for x in keys if x.endswith('fuse_mark')]
    adv_keys = [x for x in keys if x.endswith('kernel_mask')]
    for i,key_mask in enumerate(adv_keys):
        kernels = state_dict[key_mask]
        dist = 0
        for demog_comb in demog_combs:
            k1 = kernels[demog_comb[0],:,:,:].view(1,-1)
            k2 = kernels[demog_comb[1],:,:,:].view(1,-1)
            k1 = k1/torch.norm(k1,dim=1)
            k2 = k2/torch.norm(k2,dim=1)
            temp = -1.0*torch.matmul(k1, torch.transpose(k2,0,1))
            dist += temp
        dist = dist/float(len(demog_combs))
        print(dist)
        if dist < args.gac_threshold:
            state_dict[fuse_keys[i]][0] = -1
            kernels = torch.mean(kernels, dim=0).unsqueeze(0)
            kernels = kernels.repeat(len(ndemog),1,1,1)
            state_dict[key_mask] = kernels
    model.load_state_dict(state_dict)

if __name__ == "__main__":
    utils.setup_graceful_exit()
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        # do not print stack trace when ctrl-c is pressed
        pass
    except Exception:
        traceback.print_exc(file=sys.stdout)
    finally:
        traceback.print_exc(file=sys.stdout)
        utils.cleanup()


'''
GAC:
Computational complexity:       5963769856.0 5.96G
Number of parameters:           43580736.0 43.58M

Base:
Computational complexity:       10.818147328G
Number of parameters:           74.104128M 43.9998M

MAC - Multiply-Add cumulation
'''