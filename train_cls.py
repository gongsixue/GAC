# train.py

import time
import plugins
import itertools

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import losses

from itertools import combinations

import pdb

class Trainer:
    def __init__(self, args, model, criterion, evaluation, optimizer=None):
        self.args = args
        self.model = model
        self.criterion = criterion
        self.evaluation = evaluation

        self.att_loss = losses.AttMatrixCov(args.cuda)

        self.nepochs = args.nepochs

        self.lr = args.learning_rate
        self.optim_method = args.optim_method
        self.optim_options = args.optim_options
        self.scheduler_method = args.scheduler_method

        if optimizer is None:
            self.module_list = nn.ModuleList([self.model, self.criterion])
            self.optimizer = getattr(optim, self.optim_method)(
                self.module_list.parameters(), lr=self.lr, **self.optim_options)
        else:
            self.optimizer = optimizer
        if self.scheduler_method is not None:
            if self.scheduler_method != 'Customer':
                self.scheduler = getattr(optim.lr_scheduler, self.scheduler_method)(
                    self.optimizer, **args.scheduler_options)

        # for classification
        self.labels = torch.zeros(args.batch_size).long()
        self.inputs = torch.zeros(
            args.batch_size,
            args.resolution_high,
            args.resolution_wide
        )

        if args.cuda:
            self.labels = self.labels.cuda()
            self.inputs = self.inputs.cuda()

        self.inputs = Variable(self.inputs)
        self.labels = Variable(self.labels)

        # logging training
        self.log_loss = plugins.Logger(
            args.logs_dir,
            'TrainLogger.txt',
            args.save_results
        )
        params_loss = ['LearningRate','idLoss', 'attLoss']
        self.log_loss.register(params_loss)

        # monitor training
        self.monitor = plugins.Monitor()
        self.params_monitor = {
            'LearningRate': {'dtype': 'running_mean'},
            'idLoss': {'dtype': 'running_mean'},
            'attLoss': {'dtype': 'running_mean'},
        }
        self.monitor.register(self.params_monitor)

        # visualize training
        self.visualizer = plugins.Visualizer(args.port, args.env, 'Train')
        params_visualizer = {
            'LearningRate': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'learning_rate',
                    'layout': {'windows': ['train', 'test'], 'id': 0}},
            'idLoss': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'id_loss',
                    'layout': {'windows': ['train', 'test'], 'id': 0}},
            'attLoss': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'att_loss',
                    'layout': {'windows': ['train', 'test'], 'id': 0}},
            'Train_Image': {'dtype': 'image', 'vtype': 'image',
                            'win': 'train_image'},
            'Train_Images': {'dtype': 'images', 'vtype': 'images',
                             'win': 'train_images'},
        }
        self.visualizer.register(params_visualizer)

        # display training progress
        self.print_formatter = 'Train [%d/%d][%d/%d] '
        for item in params_loss:
            self.print_formatter += item + " %.4f "
        # print margin
        # self.print_formatter += "Margin %.4f %.4f %.4f %.4f"
        # self.print_formatter += "IDLoss %.4f AttLoss %.4f"

        self.losses = {}
        self.binage = torch.Tensor([10,22.5,27.5,32.5,37.5,42.5,47.5,55,75])

    def model_train(self):
        self.model.train()

    def get_kernel_dist(self):
        state_dict = self.model.state_dict()
        keys = list(state_dict)
        keys_mask = [x for x in keys if x.endswith('kernel_mask')]
        adv_keys = [x for x in keys_mask if state_dict[x].size(0) > 1]
        ndemog = self.args.ndemog
        ndemog = list(range(ndemog))
        demog_combs = list(combinations(ndemog, 2))

        dist_list = []
        for key_mask in adv_keys:
            kernels = state_dict[key_mask]
            dist = 0
            for demog_comb in demog_combs:
                k1 = kernels[demog_comb[0],:,:,:].view(1,-1)
                k2 = kernels[demog_comb[1],:,:,:].view(1,-1)
                k1 = k1/torch.norm(k1,dim=1)
                k2 = k2/torch.norm(k2,dim=1)
                # dist += torch.norm(k1-k2)
                dist += -1*torch.matmul(k1, torch.transpose(k2,0,1))
            dist = dist/float(len(ndemog))
            dist_list.append(dist.item())
        return dist_list

    def get_att_dist(self):
        state_dict = self.model.state_dict()
        keys = list(state_dict)
        adv_keys = [x for x in keys if x.endswith('att_channel')]
        ndemog = self.args.ndemog
        ndemog = list(range(ndemog))
        demog_combs = list(combinations(ndemog, 2))

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

    def train(self, epoch, dataloader, checkpoints, acc_best):
        dataloader = dataloader['train']
        self.monitor.reset()
        torch.cuda.empty_cache()

        # switch to train mode
        self.model_train()

        end = time.time()

        loss_min = 999
        stored_models = {}

        fw = open(os.path.join(self.args.logs_dir, 'dist_epoch_{}.txt'.format(epoch)), 'w')

        for i, (inputs, labels, attrs, fmetas) in enumerate(dataloader):
            # keeps track of data loading time
            data_time = time.time() - end

            ############################
            # Update network
            ############################

            batch_size = inputs.size(0)
            self.inputs.data.resize_(inputs.size()).copy_(inputs)
            self.labels.data.resize_(labels.size()).copy_(labels)
            self.labels = self.labels.view(-1)

            # attrs = attrs[1] # demographic labels
            # attrs = torch.LongTensor(attrs)
            attrs = attrs.long()
            if self.args.cuda:
                attrs = attrs.cuda()
            attrs = Variable(attrs)

            # channel attention
            outputs,attc,atts = self.model(self.inputs, attrs, epoch)
            idloss = self.criterion(outputs, self.labels)
            attloss = self.args.att_weight*self.att_loss(attc, atts)
            loss = idloss

            # margin attention
            # outputs = self.model(self.inputs)
            # loss,margin = self.criterion(outputs, self.labels, attrs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.losses['idLoss'] = idloss.item()
            self.losses['attLoss'] = attloss.item()
            for param_group in self.optimizer.param_groups:
                self.cur_lr = param_group['lr']
            self.losses['LearningRate'] = self.cur_lr
            self.monitor.update(self.losses, batch_size)

            # print batch progress
            print(self.print_formatter % tuple(
                [epoch + 1, self.nepochs, i+1, len(dataloader)]
                + [self.losses[key] for key in self.params_monitor]
                # + [m for m in margin]
                # + [idloss.item(), attloss.item()]
                ))

            # if i%10000 == 0:
            #     if self.losses['Loss'] < loss_min:
            #         loss_min == self.losses['Loss']
            #         stored_models['model'] = self.model
            #         stored_models['loss'] = self.criterion
            #         checkpoints.save(acc_best, stored_models, epoch, i, True)
            
            dist_list = self.get_kernel_dist()
            for dist in dist_list:
                fw.write(str(dist)+'\t')
            fw.write('\n')
        fw.close()
        
        # update the log file
        loss = self.monitor.getvalues()
        self.log_loss.update(loss)

        # update the visualization
        loss['Train_Image'] = inputs[0]
        loss['Train_Images'] = inputs
        self.visualizer.update(loss)

        # update the learning rate
        if self.scheduler_method is not None:
            if self.scheduler_method == 'ReduceLROnPlateau':
                self.scheduler.step(loss['idLoss'])
            elif self.scheduler_method == 'Customer':
                if epoch in self.args.lr_schedule: 
                    self.lr *= 0.1
                    self.optimizer = getattr(optim, self.optim_method)(
                        self.model.parameters(), lr=self.lr, **self.optim_options)
            else:
                self.scheduler.step()

        return self.monitor.getvalues('idLoss')
