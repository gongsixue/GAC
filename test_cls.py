# test.py

import time
import plugins
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import pdb

class Tester:
    def __init__(self, args, model, evaluation, writer):
        self.args = args
        self.model = model
        self.evaluation = evaluation
        self.writer = writer

        self.nepochs = args.nepochs

        # logging testing
        self.log_loss = plugins.Logger(
            args.logs_dir,
            'TestLogger.txt',
            args.save_results
        )
        self.params_log = ['ACC', 'STD']
        self.log_loss.register(self.params_log)

        # monitor testing
        self.monitor = plugins.Monitor()
        self.params_monitor = {}
        for param in self.params_log:
            self.params_monitor.update(
                {param: {'dtype': 'running_mean'}})
        self.monitor.register(self.params_monitor)

        # display training progress
        self.print_formatter = 'Test [%d/%d]] '
        for item in self.params_log:
            self.print_formatter += item + " %.4f "

        self.losses = {}

    def model_eval(self):
        self.model.eval()

    def test(self, dataloader, epoch):
        dataloader = dataloader['test']
        torch.cuda.empty_cache()
        self.monitor.reset()

        # switch to eval mode
        self.model_eval()

        end = time.time()

        features = []
        labels = []

        # extract query features
        for i, (inputs,input_labels,attrs,fmetas) in enumerate(dataloader):
            # keeps track of data loading time
            data_time = time.time() - end
            end = time.time()

            ############################
            # Evaluate Network
            ############################

            input_labels = input_labels.long()
            attrs = attrs.long()
            if self.args.cuda:
                inputs = inputs.cuda()
                attrs = attrs.cuda()

            if self.args.train == 'base':
                embeddings,_ = self.model(inputs)
            elif self.args.train == 'gac':
                embeddings,_ = self.model((inputs, attrs), epoch)

            feat_time = time.time() - end
            
            features.append(embeddings.data.cpu().numpy())
            labels.append(input_labels.data.numpy())

            # clear memorise of network gradients
            # pdb.set_trace()
            # torch.sum(embeddings).backward()

        labels = np.concatenate(labels, axis=0)
        features = np.concatenate(features, axis=0)
        avg,std,_ = self.evaluation(features)
        
        self.losses['ACC'] = avg
        self.losses['STD'] = std
        batch_size = 1
        self.monitor.update(self.losses, batch_size)

        # print batch progress
        print(self.print_formatter % tuple(
            [epoch + 1, self.nepochs] +
            [self.losses[key] for key in self.params_monitor]))
            
        # update the log file
        loss = self.monitor.getvalues()
        self.log_loss.update(loss)

        if self.args.save_results:
            for key in self.params_log:           
                self.writer.add_scalar(key, loss[key], epoch)

        return avg,std

    def extract_features(self, dataloader, epoch):
        dataloader = dataloader['test']
        self.model_eval()

        # extract features
        for i, (inputs,testlabels,attrs,fmetas) in enumerate(dataloader):
            attrs = attrs.long()
            if self.args.cuda:
                inputs = inputs.cuda()
                attrs = attrs.cuda()

            self.model.zero_grad()
            
            if self.args.train == 'base':
                outputs = self.model(inputs)
            elif self.args.train == 'gac':
                outputs,_,_ = self.model(inputs, attrs)

            # clear memorise of network gradients
            torch.sum(outputs).backward()
            
            if i == 0:
                embeddings = outputs.data.cpu().numpy()
                labels = testlabels.numpy()
            else:
                embeddings = np.concatenate((embeddings, outputs.data.cpu().numpy()), axis=0)
                labels = np.concatenate((labels, testlabels.numpy()), axis=0)
        
        labels = labels.reshape(-1)

        # save the features
        subdir = os.path.dirname(self.args.feat_savepath)
        if os.path.isdir(subdir) is False:
            os.makedirs(subdir)
        np.savez(self.args.feat_savepath, feat=embeddings, label=labels)
