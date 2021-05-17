# checkpoints.py

import os
import torch
from torch import nn
import torch.optim as optim
import pickle

import pdb

class Checkpoints:
    def __init__(self, args):
        self.args = args
        self.dir_save = args.save_dir
        self.model_filename = args.resume
        self.save_results = args.save_results
        self.cuda = args.cuda

        if self.save_results and not os.path.isdir(self.dir_save):
            os.makedirs(self.dir_save)

    def latest(self, name):
        if name == 'resume':
            return self.model_filename

    def save(self, acc, models, epoch):
        keys = list(models)
        assert(len(keys) == 3)
        filename = '%s/model_epoch_%d_%.6f.pth.tar' % (self.dir_save, epoch, acc)

        checkpoint = {}

        # save model
        checkpoint['model'] = models['model'].state_dict()

        # save criterion
        keys = list(models['loss'])
        for key in keys:
            newkey = 'loss_'+key
            checkpoint[newkey] = models['loss'][key].state_dict()
            
        # save optimizer
        checkpoint['optimizer'] = models['optimizer'].state_dict()

        torch.save(checkpoint, filename)

    def load(self, models, filename, old=False):
        if old:
            filename_model = os.path.join(filename, 'model_epoch_25_final_95.1196248.pth')
            filename_loss = os.path.join(filename, 'loss_epoch_25_final_95.1196248.pkl')

            model = models['model']['classify']
            if os.path.exits(filename_model) and os.path.exists(filename_loss):
                if filename_model.endswith('pth'):
                    state_dict = torch.load(filename_model)
                    saved_params = list(state_dict)
                    
                    update_dict = {}
                    model_params = list(model.state_dict())
                    for i,key in model_params:
                        update_dict[key] = state_dict[saved_params[i]]
                    models['model']['classify'].load_state_dict(update_dict)
                elif filename_model.endswith('pkl'):
                    with open(filename_model, 'rb') as f:
                        saved_model = pickle.load(f)
                    keys = list(saved_model)
                    for key in keys:
                        models['model'][key].load_state_dict(saved_model[key], strict=False)
                
                with open(filename_loss, 'rb') as f:
                    saved_loss = pickle.load(f)
                keys = list(saved_loss)
                for key in keys:
                    models['loss'][key].load_state_dict(saved_loss[key], strict=False)

                return models
        
        else:
            if os.path.isfile(filename):
                print("=> loading checkpoint '{}'".format(filename))
                if self.cuda:
                    checkpoint = torch.load(filename)
                else:
                    checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
                
                keys_check = list(checkpoint)
                for key_check in keys_check:
                    if key_check.startswith('model'):
                        models['model'].load_state_dict(checkpoint[key_check])
                    elif key_check.startswith('loss'):
                        key = key_check.split('_')[1]
                        models['loss'][key].load_state_dict(checkpoint[key_check])
                    elif key_check.startswith('optimizer'):
                        models['optimizer'].load_state_dict(checkpoint[key_check])
                return models
            else:
                raise (Exception("=> no checkpoint found at '{}'".format(filename)))
