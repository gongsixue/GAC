import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.autograd import Variable

import models

import pdb

def remakeImage(img):
    i_min = np.min(img)
    i_max = np.max(img)

    k = 255.0/float(i_max - i_min)
    b = 255.0*i_min/float(i_min - i_max)
    img = k*img + b
    img = np.uint8(img)
    created_image = Image.fromarray(img, 'RGB')
    return created_image

class ImageLoader():
    def __init__(self, path_pic, if_cuda=True):
        img = Image.open(path_pic).convert('RGB')
        preprocess = transforms.Compose([transforms.CenterCrop((112,112)),
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        self.img = preprocess(img)
        self.img = self.img.unsqueeze(0)

        self.if_cuda = if_cuda

    def preprocess(self):
        if self.if_cuda:
            self.img = self.img.cuda()

        self.img = Variable(self.img, requires_grad=True)

class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, path_model, prefix, ngpu=1, if_cuda=True):
        self.prefix = prefix
        self.if_cuda = if_cuda

        use_spatial_att = False
        nclasses = 28000
        model_options = {"nclasses": nclasses}
        self.model = models.subbase_face50(**model_options)
        # self.model = nn.DataParallel(self.model, device_ids=list(range(ngpu)))
        if if_cuda:
            self.model = self.model.cuda()
        
        model_dict = self.model.state_dict()
        state_dict = torch.load(path_model)
        update_dict = {}
        valid_keys = list(model_dict)
        state_keys = list(state_dict)
        state_ind = 0
        for key in valid_keys:
            # if key.endswith('num_batches_tracked'):
            #     continue
            update_dict[key] = state_dict[state_keys[state_ind]]
            state_ind += 1 
        self.model.load_state_dict(update_dict)
        self.model = self.model.eval()

        self.conv_output = 0

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]
        # Hook the selected layer
        self.model.register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self, img, selected_layer, selected_filter):
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter

        # Hook the selected layer
        self.hook_layer()
        # Define optimizer for the image
        optimizer = torch.optim.Adam([img], lr=0.1, weight_decay=1e-6)
        for i in range(1, 31):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = img
            x = self.model(x)
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.item()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            output = img.data.cpu().numpy()[0].transpose(1,2,0)
            self.created_image = remakeImage(output)
            # Save image
            if i % 5 == 0:
                im_path = self.prefix + 'feathook_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                self.created_image.save(im_path)

    def visualise_layer_without_hooks(self, img, selected_layer, selected_filter):
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter

        # Process image and return variable
        # Define optimizer for the image
        optimizer = torch.optim.Adam([img], lr=0.1, weight_decay=1e-6)
        for i in range(1, 31):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = img
            x = self.model(x)
            self.conv_output = x[0, self.selected_filter]
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.item()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            output = img.data.cpu().numpy()[0].transpose(1,2,0)
            self.created_image = remakeImage(output)
            # Save image
            if i % 5 == 0:
                im_path = self.prefix + 'feathook_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                self.created_image.save(im_path)

def main():
    if_cuda = True
    ngpu = 1
    demog_label = [3]
    selected_layer = 39
    selected_filter = 253

    savepath_pic = '/research/prip-gongsixu/codes/biasface/results/figures/neurIPS2020/visual/filter/base/ifs/'
    if os.path.isdir(savepath_pic) is False:
        os.makedirs(savepath_pic)
    
    path_pic = '/research/prip-gongsixu/codes/biasface/results/figures/neurIPS2020/visual/if_sample.jpg'
    folder_model = '/research/prip-gongsixu/codes/biasface/results/models/attention/base/balance_base50/Save'
    path_model = os.path.join(folder_model,'model_epoch_25_final_0.996333.pth')

    loader = ImageLoader(path_pic, if_cuda)
    loader.preprocess()

    visualizer = CNNLayerVisualization(path_model, savepath_pic, ngpu, if_cuda)
    visualizer.visualise_layer_with_hooks(loader.img, selected_layer, selected_filter)

if __name__ == '__main__':
    main()