# train.py

import time
import plugins
import itertools
from random import shuffle
import math

import os

import torch
import torch.optim as optim
from torch.autograd import Variable

import losses

import pdb

class Trainer:
    def __init__(self, args, model, criterion, evaluation, optimizer, writer):
        self.args = args
        self.model = model
        self.criterion = criterion
        self.evaluation = evaluation
        self.optimizer = optimizer
        self.writer = writer

        self.nepochs = args.nepochs

        self.lr = args.learning_rate
        self.optim_method = args.optim_method
        self.optim_options = args.optim_options
        self.scheduler_method = args.scheduler_method

        self.att_loss = losses.AttMatrixCov(args.cuda)
        self.dist_loss = losses.DebiasIntraDist(args.cuda)
        
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
        self.params_loss = ['LearningRate','idLoss', 'attLoss']
        self.log_loss.register(self.params_loss)

        # monitor training
        self.monitor = plugins.Monitor()
        self.params_monitor = {
            'LearningRate': {'dtype': 'running_mean'},
            'idLoss': {'dtype': 'running_mean'},
            'attLoss': {'dtype': 'running_mean'},
        }
        self.monitor.register(self.params_monitor)

        # display training progress
        self.print_formatter = 'Train [%d/%d][%d/%d] '
        for item in self.params_loss:
            self.print_formatter += item + " %.4f "
        # self.print_formatter += "IDLoss %.4f AttLoss %.4f"

        self.losses = {}

    def model_train(self):
        self.model.train()

    def train(self, epoch, dataloader, checkpoints, acc_best):
        dataloader = dataloader['train']
        self.monitor.reset()

        # switch to train mode
        self.model_train()

        end = time.time()

        index_list = list(range(len(dataloader)))
        shuffle(index_list)
        num_batch = math.ceil(float(len(index_list))/float(self.args.batch_size))
        for i in range(num_batch):
            # keeps track of data loading time
            data_time = time.time() - end

            start_index = i*self.args.batch_size
            end_index = min((i+1)*self.args.batch_size, len(index_list))
            batch_size = end_index - start_index
            data = []

            # iter_info = []
            for j,ind in enumerate(index_list[start_index:end_index]):
                rawdata = dataloader[ind]
                
                if j == 0:
                    inputs = rawdata[0]
                    labels = rawdata[1]
                    attributes = rawdata[2]
                    fmetas = rawdata[3]
                else:
                    inputs = torch.cat((inputs, rawdata[0]), 0)
                    labels = torch.cat((labels, rawdata[1]), 0)
                    attributes = torch.cat((attributes, rawdata[2]), 0)
                    fmetas.extend(rawdata[3])

            ############################
            # Update network
            ############################

            attrs = torch.squeeze(attributes, dim=1)
            labels = torch.squeeze(labels, dim=1)

            labels = labels.long()
            attrs = attrs.long()
            if self.args.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
                attrs = attrs.cuda()

            # channel attention
            # outputs,attc,atts = self.model(inputs, attrs)
            outputs = self.model(inputs)
            idloss = self.criterion['idLoss'](outputs, labels)
            # attloss = self.args.att_weight*self.att_loss(attc, atts)
            distloss = self.dist_loss(outputs,labels,attrs)
            loss = idloss + 0.1*distloss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.losses['idLoss'] = idloss.item()
            self.losses['attLoss'] = distloss.item()
            for param_group in self.optimizer.param_groups:
                self.cur_lr = param_group['lr']
            self.losses['LearningRate'] = self.cur_lr

            self.monitor.update(self.losses, batch_size)

            # print batch progress
            print(self.print_formatter % tuple(
                [epoch + 1, self.nepochs, i+1, num_batch] +
                [self.losses[key] for key in self.params_monitor]
                # + [idloss.item(), attloss.item()]
                ))

            # update tensor board
            if i%1000 == 0:
                loss = self.monitor.getvalues()
                for key in self.params_loss:          
                    self.writer.add_scalar(key, loss[key], len(dataloader)*epoch+i)

        # update the log file
        loss = self.monitor.getvalues()
        test_freq = 45
        if float(epoch) % test_freq == 0:
            self.log_loss.update(loss)

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
