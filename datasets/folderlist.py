# folderlist.py

import os
import math
import pickle
import os.path
import utils as utils
import torch.utils.data as data
from sklearn.utils import shuffle
import datasets.loaders as loaders

import pdb

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(ifolder):
    subfolders = os.listdir(ifolder)

    images = []
    labels = []

    for subfolder in subfolders:
        subpath = os.path.join(ifolder, subfolder)
        if os.path.isdir(subpath):
            images = os.listdir(subpath)
            for image in images:
                if image.endswith('jpg'):
                    images.append((os.path.join(subpath,image), subfolder))
                    labels.append(subfolder)

    labels = list(set(labels))

    return images, labels


class FolderListLoader(data.Dataset):
    def __init__(self, ifile, transform=None,
                 loader='image'):

        imagelist, labellist = make_dataset(ifile)
        if len(imagelist) == 0:
            raise(RuntimeError("No images found"))
        if len(labellist) == 0:
            raise(RuntimeError("No labels found"))

        if loader == 'image':
            self.loader = loaders.loader_image
        if loader == 'torch':
            self.loader = loaders.loader_torch
        if loader == 'numpy':
            self.loader = loaders.loader_numpy

        self.transform = transform

        self.images = imagelist
        self.labels = labellist

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if len(self.images) > 0:
            path = self.images[index][0]
            image = self.loader(path)
        else:
            image = []

        if len(self.labels) > 0:
            label = self.labels.index[self.images[index][1]]
        else:
            label = 0

        if self.transform is not None:
            image = self.transform(image)

        return image, label
