# checkpoints.py

import os
import torch
from torch import nn
import torch.optim as optim

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

    def save(self, acc, models, epoch, step, best):
        keys = list(models)
        assert(len(keys) == 3)
        if best is True:
            torch.save(models['model'].state_dict(),
                       '%s/model_epoch_%d_%s_%.6f.pth' % (self.dir_save, epoch, str(step), acc))
            torch.save(models['loss'].state_dict(),
                       '%s/loss_epoch_%d_%s_%.6f.pth' % (self.dir_save, epoch, str(step), acc))
            # torch.save(models['model'].state_dict(),
            #            '%s/optimizer_epoch_%d_%s_%.6f.pth' % (self.dir_save, epoch, str(step), acc))

    def load(self, models, filename):
        if os.path.isfile(filename):
            model = models['model']
            print("=> loading checkpoint '{}'".format(filename))
            if self.cuda:
                state_dict = torch.load(filename)
            else:
                state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
            model_dict = model.state_dict()
            update_dict = {}
            valid_keys = list(model_dict)
            state_keys = list(state_dict)
            state_ind = 0
            for key in valid_keys:
                # if key.endswith('num_batches_tracked'):
                #     continue
                update_dict[key] = state_dict[state_keys[state_ind]]
                state_ind += 1 
            model.load_state_dict(update_dict)
            models['model'] = model
            return models

        elif os.path.isdir(filename):
            filename_model = os.path.join(filename, 'model_epoch_22_final_0.943167.pth')
            filename_loss = os.path.join(filename, 'model_epoch_22_final_0.943167.pth')
            # filename_optimizer = os.path.join(filename, 'optimizer_epoch_26_final_0.996000.pth')

            state_dict = torch.load(filename_model)
            saved_params = list(state_dict)

            models['model'].load_state_dict(state_dict)
            models['loss'].load_state_dict(torch.load(filename_loss), strict=False)
            # models['optimizer'] = None
            # module_list = nn.ModuleList([models['model'], models['loss']])
            # models['optimizer'] = getattr(optim, self.args.optim_method)(
            #     module_list.parameters(), lr=self.args.learning_rate, **self.args.optim_options)
            # models['optimizer'].load_state_dict(torch.load(filename_optimizer))

            return models
        raise (Exception("=> no checkpoint found at '{}'".format(filename)))
