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
    def __init__(self, args, model, criterion, evaluation):
        self.args = args
        self.model = model
        self.criterion = criterion
        self.evaluation = evaluation

        self.nepochs = args.nepochs

        self.module_list = nn.ModuleList([self.model, self.criterion])
        self.optimizer = getattr(optim, args.optim_method)(
            self.module_list.parameters(), lr=args.learning_rate, **args.optim_options)

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

        # logging testing
        self.log_loss = plugins.Logger(
            args.logs_dir,
            'TestLogger.txt',
            args.save_results
        )
        params_loss = ['Test_Result']
        self.log_loss.register(params_loss)

        # monitor testing
        self.monitor = plugins.Monitor()
        self.params_monitor = {
            params_loss[0]: {'dtype': 'running_mean'},
        }
        self.monitor.register(self.params_monitor)

        # visualize testing
        self.visualizer = plugins.Visualizer(args.port, args.env, 'Test')
        params_visualizer = {
            params_loss[0]: {'dtype': 'scalar', 'vtype': 'plot', 'win': 'acc',
                    'layout': {'windows': ['train', 'test'], 'id': 1}},
            # 'Test_Image': {'dtype': 'image', 'vtype': 'image',
            #                'win': 'test_image'},
            # 'Test_Images': {'dtype': 'images', 'vtype': 'images',
            #                 'win': 'test_images'},
        }
        self.visualizer.register(params_visualizer)

        # display training progress
        self.print_formatter = 'Test [%d/%d]] '
        for item in [params_loss[0]]:
            self.print_formatter += item + " %.4f "

        self.losses = {params_loss[0]:0.0}
        # self.binage = torch.Tensor([19,37.5,52.5,77])

    def model_eval(self):
        self.model.eval()

    def test(self, epoch, dataloader):
        dataloader = dataloader['test']
        self.monitor.reset()
        # torch.cuda.empty_cache()

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

            self.inputs.data.resize_(inputs.size()).copy_(inputs)
            self.labels.data.resize_(input_labels.size()).copy_(input_labels)

            # attrs = attrs[1] # demographic labels
            attrs = attrs.long()
            if self.args.cuda:
                attrs = attrs.cuda()
            attrs = Variable(attrs)

            embeddings,_,_ = self.model(self.inputs, attrs, epoch)
            # embeddings = self.model(self.inputs)

            feat_time = time.time() - end
            
            features.append(embeddings.data.cpu().numpy())
            labels.append(input_labels.data.numpy())

            torch.sum(embeddings).backward()

        labels = np.concatenate(labels, axis=0)
        features = np.concatenate(features, axis=0)
        results,mean,_ = self.evaluation(features)
        
        self.losses[list(self.losses)[0]] = results
        batch_size = 1
        self.monitor.update(self.losses, batch_size)

        # print batch progress
        print(self.print_formatter % tuple(
            [epoch + 1, self.nepochs] +
            [results]))
            
        # update the log file
        loss = self.monitor.getvalues()
        self.log_loss.update(loss)

        # update the visualization
        self.visualizer.update(loss)

        # np.savez('/research/prip-gongsixu/codes/biasface/results/model_analysis/result_agel20.npz',
        #     preds=preds.cpu().numpy(), labels=labels.cpu().numpy())

        return results,mean

    def extract_features(self, dataloader, epoch):
        dataloader = dataloader['test']
        self.model_eval()

        end = time.time()

        # extract features
        filenames = []
        for i, (inputs,testlabels,attrs,fmetas) in enumerate(dataloader):

            self.inputs.data.resize_(inputs.size()).copy_(inputs)

            # attrs = attrs[1] # demographic labels
            # attrs = torch.LongTensor(attrs)
            attrs = attrs.long()
            if self.args.cuda:
                attrs = attrs.cuda()
            attrs = Variable(attrs)

            self.model.zero_grad()
            # outputs = self.model(self.inputs)
            outputs,_,_ = self.model(self.inputs, attrs)
            # outputs,_,_ = self.model(self.inputs, attrs, epoch)

            torch.sum(outputs).backward()
            
            if i == 0:
                embeddings = outputs.data.cpu().numpy()
                labels = testlabels.numpy()
            else:
                embeddings = np.concatenate((embeddings, outputs.data.cpu().numpy()), axis=0)
                labels = np.concatenate((labels, testlabels.numpy()), axis=0)

        data_time = time.time() - end
        print(data_time)
        
        labels = labels.reshape(-1)

        # save the features
        subdir = os.path.dirname(self.args.feat_savepath)
        if os.path.isdir(subdir) is False:
            os.makedirs(subdir)
        np.savez(self.args.feat_savepath, feat=embeddings, label=labels)
        # with open(os.path.splitext(self.args.feat_savepath)[0]+'_list.txt','w') as f:
        #     for filename in filenames:
        #         f.write(filename+'\n')
