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

from itertools import combinations

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

    # The trainer handles the training loop
    trainer = Trainer(args, model, model_dict['loss'], evaluation, model_dict['optimizer'])
    # The trainer handles the evaluation on validation set
    tester = Tester(args, model, model_dict['loss'], evaluation)

    test_freq = 1

    dataloader = Dataloader(args)

    if args.extract_feat:
        loaders  = dataloader.create(flag='Test')
        tester.extract_features(loaders, 1)
    elif args.just_test:
        loaders  = dataloader.create(flag='Test')
        acc_test,acc_mean = tester.test(args.epoch_number, loaders, 1)
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
            loss_train = trainer.train(epoch, loaders, checkpoints, acc_best)
            #### fuse kernels ####
            if epoch == args.fuse_epoch:
                state_dict = model.state_dict()
                keys = list(state_dict)
                keys_mask = [x for x in keys if x.endswith('kernel_mask')]
                adv_keys = [x for x in keys_mask if state_dict[x].size(0) > 1]
                # temp_keys = [x for x in adv_keys if torch.sum(state_dict[x][1,:,:,:]-state_dict[x][2,:,:,:]) > 0]
                # print(len(temp_keys))
                for key_mask in adv_keys:
                    kernels = state_dict[key_mask]
                    dist = 0
                    for demog_comb in demog_combs:
                        k1 = kernels[demog_comb[0],:,:,:].view(1,-1)
                        k2 = kernels[demog_comb[1],:,:,:].view(1,-1)
                        k1 = k1/torch.norm(k1,dim=1)
                        k2 = k2/torch.norm(k2,dim=1)
                        dist += -1*torch.matmul(k1, torch.transpose(k2,0,1))
                    dist = dist/float(len(demog_combs))
                    if dist <= args.gac_threshold:
                        kernels = torch.mean(kernels, dim=0).unsqueeze(0)
                        kernels = kernels.repeat(len(ndemog),1,1,1)
                        state_dict[key_mask] = kernels
                model.load_state_dict(state_dict)
            #### fuse kernels ####

            if float(epoch) % test_freq == 0:
                acc_test,acc_mean = tester.test(epoch, loaders)

            if loss_best > loss_train:
                model_best = True
                loss_best = loss_train
                acc_best = acc_test
            model_best = True
            if  float(epoch) % test_freq == 0 and args.save_results:
                stored_models['model'] = model
                stored_models['loss'] = trainer.criterion
                stored_models['optimizer'] = trainer.optimizer
                checkpoints.save(acc_test, stored_models, epoch, 'final', model_best)

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
