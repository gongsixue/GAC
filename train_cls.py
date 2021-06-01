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
    def __init__(self, args, model, criterion, optimizer, writer):
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.writer = writer

        self.att_loss = losses.AttMatrixCov(args.cuda)
        self.debias_loss = losses.DebiasIntraDist(args.cuda)
        self.deb_weight = args.deb_weight

        self.nepochs = args.nepochs

        self.lr = args.learning_rate
        self.scheduler_method = args.scheduler_method
        if self.scheduler_method is not None:
            if self.scheduler_method != 'Customer':
                self.scheduler = getattr(optim.lr_scheduler, self.scheduler_method)(
                    self.optimizer, **args.scheduler_options)

        # logging training
        self.log_loss = plugins.Logger(
            args.logs_dir,
            'TrainLogger.txt',
            args.save_results
        )
        self.params_log = ['LearningRate','idLoss', 'debLoss']
        self.log_loss.register(self.params_log)

        # monitor training
        self.monitor = plugins.Monitor()
        self.params_monitor = {}
        for param in self.params_log:
            self.params_monitor.update(
                {param: {'dtype': 'running_mean'}})
        self.monitor.register(self.params_monitor)

        # display training progress
        self.print_formatter = 'Train [%d/%d][%d/%d] '
        for item in self.params_log:
            self.print_formatter += item + " %.4f "

        self.losses = {}

    def model_train(self):
        self.model.train()

    def get_adavk_dist(self):
        state_dict = self.model.state_dict()
        keys = list(state_dict)
        fuse_keys = [x for x in keys if x.endswith('fuse_mark')]
        adv_keys = [x for x in keys if x.endswith('kernel_mask')]
        ndemog = self.args.ndemog
        ndemog = list(range(ndemog))
        demog_combs = list(combinations(ndemog, 2))

        dist_list = []
        for i,key_mask in enumerate(adv_keys):
            kernels = state_dict[key_mask]
            fuse_mark = state_dict[fuse_keys[i]]
            if fuse_mark == -1:
                continue            
            dist = 0
            for demog_comb in demog_combs:
                k1 = kernels[demog_comb[0],:,:,:].view(1,-1)
                k2 = kernels[demog_comb[1],:,:,:].view(1,-1)
                k1 = k1/torch.norm(k1,dim=1)
                k2 = k2/torch.norm(k2,dim=1)
                dist += -1*torch.matmul(k1, torch.transpose(k2,0,1))
            dist = dist/float(len(demog_combs))
            dist_list.append(dist.item())
        
        print(len(dist_list))
        return dist_list

    def print_grad(self):
        parameters = list(self.model.named_parameters())
        indices = [x[0] for x in enumerate(parameters) if x[1][0].endswith('kernel_mask')]

        kernels_grad = parameters[indices[-1]][1].grad
        print(kernels_grad[0,0,0,0], kernels_grad[1,0,0,0], kernels_grad[2,0,0,0])

    def get_num_adavk(self):
        state_dict = self.model.state_dict()
        keys = list(state_dict)
        fuse_keys = [x for x in keys if x.endswith('fuse_mark')]
        num_adavk = 0
        num_norm = 0
        for key in fuse_keys:
            fuse_mark = state_dict[key]
            if fuse_mark == 0:
                num_norm += 1
            if fuse_mark == -1:
                num_adavk += 1
        return num_adavk, num_norm

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

    def train(self, dataloader, epoch):
        dataloader = dataloader['train']
        self.monitor.reset()
        torch.cuda.empty_cache()

        # switch to train mode
        self.model_train()

        end = time.time()

        if self.args.save_adaptive_distance:
            f_kernel = open(os.path.join(self.args.logs_dir, 'kernel_epoch_{}.txt'.format(epoch)), 'w')
            f_att = open(os.path.join(self.args.logs_dir, 'att_epoch_{}.txt'.format(epoch)), 'w')

        for i, (inputs, labels, attrs, fmetas) in enumerate(dataloader):
            # keeps track of data loading time
            data_time = time.time() - end

            ############################
            # Update network
            ############################

            batch_size = inputs.size(0)

            labels = labels.long()
            attrs = attrs.long()
            if self.args.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
                attrs = attrs.cuda()

            if self.args.train == 'base':
                outputs,_ = self.model(inputs)
                loss = self.criterion['idLoss'](outputs, labels)
                self.losses['idLoss'] = loss.item()
                self.losses['debLoss'] = 0
            elif self.args.train == 'gac':
                outputs,attention = self.model((inputs, attrs), epoch)
                idloss = self.criterion['idLoss'](outputs, labels)
                debloss = self.debias_loss(outputs,labels,attrs)
                attloss = self.att_loss(attention['attc'], attention['atts'])
                loss = idloss + self.deb_weight*debloss + self.args.att_weight*attloss
                self.losses['idLoss'] = idloss.item()
                self.losses['debLoss'] = debloss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            for param_group in self.optimizer.param_groups:
                self.cur_lr = param_group['lr']
            self.losses['LearningRate'] = self.cur_lr
            self.monitor.update(self.losses, batch_size)

            # print batch progress
            print(self.print_formatter % tuple(
                [epoch + 1, self.nepochs, i+1, len(dataloader)]
                + [self.losses[key] for key in self.params_monitor]
                # + [m for m in margin]
                ))

            # update tensor board
            if self.args.save_results:
                if i%1000 == 0:
                    loss = self.monitor.getvalues()
                    for key in self.params_log:          
                        self.writer.add_scalar(key, loss[key], len(dataloader)*epoch+i)

            if self.args.save_adaptive_distance:
                kernel_list = self.get_kernel_dist()
                att_list = self.get_att_dist()
                for kernel in kernel_list:
                    f_kernel.write(str(kernel)+'\t')
                f_kernel.write('\n')

                for att in att_list:
                    f_att.write(str(att)+'\t')
                f_att.write('\n')

        if self.args.save_adaptive_distance:
            f_kernel.close()
            f_att.close()
        
        # update the log file
        loss = self.monitor.getvalues()
        self.log_loss.update(loss)

        # update the learning rate
        if self.scheduler_method is not None:
            if self.scheduler_method == 'ReduceLROnPlateau':
                self.scheduler.step(loss['idLoss'])
            elif self.scheduler_method == 'Customer':
                if epoch in self.args.lr_schedule: 
                    self.lr *= 0.1
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lr
            else:
                self.scheduler.step()

        return self.monitor.getvalues('idLoss')

def update_kernels(args, model, demog_combs):
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
        if dist <= args.gac_threshold:
            state_dict[fuse_keys[i]][0] = -1
            temp = torch.mean(kernels, dim=0).unsqueeze(0)
            for i in range(args.ndemog):
                kernels[i,:,:,:] = temp
            state_dict[key_mask] = kernels
    model.load_state_dict(state_dict)