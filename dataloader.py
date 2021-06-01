# dataloader.py

import os

import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from datasets import *

__all__ = ['Dataloader']

class Dataloader:
    def __init__(self, args):
        self.args = args

    def setup(self, dataloader_type, dataset_options):        
        if dataloader_type == 'FileListLoader':
            dataset = FileListLoader(**dataset_options)
        elif dataloader_type == 'FolderListLoader':
            dataset = FolderListLoader(**dataset_options)
        elif dataloader_type == 'CSVListLoader':
            dataset = CSVListLoader(**dataset_options)
        elif dataloader_type == 'ClassSamplesDataLoader':
            dataset = ClassSamplesDataLoader(**dataset_options)
        elif dataloader_type == 'GenderCSVListLoader':
            dataset = GenderCSVListLoader(**dataset_options)
        elif dataloader_type == 'AgeBinaryLoader':
            dataset = AgeBinaryLoader(**dataset_options)
        elif dataloader_type == 'H5pyLoader':
            dataset = H5pyLoader(**dataset_options)        
        elif dataloader_type is None:
            print("No data assigned!")
        else:
            raise(Exception("Unknown Training Dataset"))

        return dataset

    def create(self, dataset=None, flag=None):
        dataloader = {}
        if dataset is not None:
            dataloader['test'] = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.args.test_batch_size,
                num_workers=int(self.args.nthreads),
                shuffle=False,pin_memory=True
            )
            return dataloader
            
        else:
            if flag == "Train":
                self.dataset_train = self.setup(self.args.dataset_train, self.args.dataset_options_train)
                dataloader['train'] = torch.utils.data.DataLoader(
                    self.dataset_train,
                    batch_size=self.args.batch_size,
                    num_workers=int(self.args.nthreads),
                    shuffle=True,pin_memory=True
                )
                return dataloader

            elif flag == "Test":
                self.dataset_test = self.setup(self.args.dataset_test, self.args.dataset_options_test)
                dataloader['test'] = torch.utils.data.DataLoader(
                    self.dataset_test,
                    batch_size=self.args.test_batch_size,
                    num_workers=int(self.args.nthreads),
                    shuffle=False,pin_memory=True
                )
                return dataloader

            elif flag is None:
                self.dataset_train = self.setup(self.args.dataset_train, self.args.dataset_options_train)
                self.dataset_test = self.setup(self.args.dataset_test, self.args.dataset_options_test)
                dataloader['train'] = torch.utils.data.DataLoader(
                    self.dataset_train,
                    batch_size=self.args.batch_size,
                    num_workers=int(self.args.nthreads),
                    shuffle=True,pin_memory=True
                )
                dataloader['test'] = torch.utils.data.DataLoader(
                    self.dataset_test,
                    batch_size=self.args.test_batch_size,
                    num_workers=int(self.args.nthreads),
                    shuffle=False,pin_memory=True
                )
                return dataloader