import os
import copy
import numpy as np
from PIL import Image
import matplotlib.cm as mpl_color_map
import matplotlib.colors as colors

import torch
import torch.nn as nn
from torch.nn import PReLU
from torch.nn import Dropout
import torch.optim as optim

import torchvision.transforms as transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt

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

def get_positive_negative_saliency(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize
    returns:
        pos_saliency ( )
    """
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency

def save_class_activation_images(org_img, activation_map, file_name):
    """
        Saves cam activation map and activation map on the original image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists('../results'):
        os.makedirs('../results')
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'hsv')
    # Save colored heatmap
    path_to_file = os.path.join('../results', file_name+'_Cam_Heatmap.png')
    save_image(heatmap, path_to_file)
    # Save heatmap on iamge
    path_to_file = os.path.join('../results', file_name+'_Cam_On_Image.png')
    save_image(heatmap_on_image, path_to_file)
    # SAve grayscale heatmap
    path_to_file = os.path.join('../results', file_name+'_Cam_Grayscale.png')
    save_image(activation_map, path_to_file)

def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image

def load_image(path_pic, if_cuda=True):
    img = Image.open(path_pic).convert('RGB')
    preprocess = transforms.Compose([transforms.CenterCrop((112,112)),
        transforms.Resize((112,112)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    img = preprocess(img)
    img = img.unsqueeze(0)

    if if_cuda:
        img = img.cuda()

    img = Variable(img, requires_grad=True)
    return img

def set_model(path_model, if_cuda=True):
    use_spatial_att = False
    nclasses = 28000
    model_options = {"use_spatial_att":use_spatial_att, "nclasses": nclasses, "ndemog":4,\
        "hard_att_channel":False, "hard_att_spatial":False, \
        "lowresol_set":{'rate':1.0, 'mode':'nearest'}}
    model = models.subnet_face50(**model_options)
    # model = nn.DataParallel(model, device_ids=list(range(ngpu)))
    if if_cuda:
        model = model.cuda()
    
    model_dict = model.state_dict()
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
    model.load_state_dict(update_dict)
    model = model.eval()
    return model

class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model, if_cuda=True):
        self.model = model
        self.if_cuda = if_cuda

        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()

        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        first_layer = list(self.model._modules.items())[1][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = ()
            for i in range(len(grad_in)):
                if i == 0:
                    temp = corresponding_forward_output * torch.clamp(grad_in[i], min=0.0)
                else:
                    temp = grad_in[i]
                modified_grad_out = modified_grad_out + (temp,)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return modified_grad_out

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            if isinstance(ten_out,dict):
                ten_out = ten_out['x']
            if isinstance(ten_out,tuple):
                ten_out = ten_out[0]
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model._modules.items():
            # if isinstance(module, PReLU):
            #     continue
            if pos == 'maxpool' or 'bn1':
                continue
            if pos == 'attconv1':
                continue
            if pos == 'bn4':
                break
            module.register_backward_hook(relu_backward_hook_function)
            module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, target_class, demog_label):
        demog_label = torch.LongTensor(demog_label)
        if self.if_cuda:
            demog_label = demog_label.cuda()
        demog_label = Variable(demog_label)

        # Forward pass
        model_output = self.model(input_image, demog_label)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        # one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output = torch.ones(1, model_output.size()[-1])
        # one_hot_output[0][target_class] = 1
        if self.if_cuda:
            one_hot_output = one_hot_output.cuda()
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # torch.sum(model_output).backward()
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.cpu().numpy()[0]
        
        return gradients_as_arr

class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer, demog_label, if_cuda=True):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.if_cuda = if_cuda

        demog_label = torch.LongTensor(demog_label)
        if self.if_cuda:
            demog_label = demog_label.cuda()
        demog_label = Variable(demog_label)
        self.demog_label = demog_label

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        nlayer = 0
        for module_pos, module in self.model._modules.items():
            if nlayer == 0:
                x,_,_ = module(x, self.demog_label)
            elif nlayer < 5:
                x = module(x)
            elif nlayer == 5:
                x,_,_ = module(x, self.demog_label)
            elif nlayer == 6:
                x_dict = {'x':x, 'demog_label':self.demog_label}
                x_dict = module(x_dict)
                x = x_dict['x']
            elif nlayer > 6 and nlayer <= 9:              
                x_dict = module(x_dict)
                x = x_dict['x']
            elif nlayer == 10:
                x = module(x)
            elif nlayer == 11:
                x,_,_ = module(x, self.demog_label)
            elif nlayer > 11 and nlayer < 13:
                x = module(x)
            elif nlayer == 13:
                x = x.view(x.size(0), -1)
                x = module(x)
            else:
                x = module(x)

            nlayer += 1
            if nlayer == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        return conv_output, x

class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer, demog_label, if_cuda=True):
        self.if_cuda = if_cuda
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer, demog_label, if_cuda)

    def generate_cam(self, input_image, target_class=0):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        # Target for backprop
        # one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output = torch.ones(1, model_output.size()[-1])
        # one_hot_output[0][target_class] = 1
        if self.if_cuda:
            one_hot_output = one_hot_output.cuda()
        # Zero grads
        self.model.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.cpu().numpy()[0]
        # Get convolution outputs
        target = conv_output.data.cpu().numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.

        # You can also use the code below instead of the code line above, suggested by @ ptschandl
        # from scipy.ndimage.interpolation import zoom
        # cam = zoom(cam, np.array(input_image[0].shape[1:])/np.array(cam.shape))
        return cam

def guided_grad_cam(grad_cam_mask, guided_backprop_mask):
    """
        Guided grad cam is just pointwise multiplication of cam mask and
        guided backprop mask

    Args:
        grad_cam_mask (np_arr): Class activation map mask
        guided_backprop_mask (np_arr):Guided backprop mask
    """
    grad_cam_mask = grad_cam_mask[None,:,:]
    grad_cam_mask = np.repeat(grad_cam_mask, 3, axis=0)
    cam_gb = np.multiply(grad_cam_mask, guided_backprop_mask)
    return cam_gb

def grad_cam(pretrained_model, ori_img, prep_img, if_cuda, demog_label, 
    target_class, target_layer, savepath_heat, savepath_heatimg, savepath_gcam):
    # Grad cam
    gcv2 = GradCam(pretrained_model, target_layer, demog_label)
    # Generate cam mask
    cam = gcv2.generate_cam(prep_img, target_class)
    print('Grad cam completed')

    # Guided backprop
    GBP = GuidedBackprop(pretrained_model, if_cuda)
    # Get gradients
    guided_grads = GBP.generate_gradients(prep_img, target_class, demog_label)
    print('Guided backpropagation completed')

    heatmap, heatmap_on_image = apply_colormap_on_image(ori_img, cam, 'hsv')
    heatmap = heatmap.convert('RGB')
    heatmap_on_image = heatmap_on_image.convert('RGB')
    heatmap.save(savepath_heat)
    heatmap_on_image.save(savepath_heatimg)

    # plt.imshow(cam, cmap='jet', interpolation='nearest')
    # plt.axis('off')
    # plt.savefig(savepath_heat, bbox_inches='tight')

    # Guided Grad cam
    cam_gb = guided_grad_cam(cam, guided_grads)
    new_img = cam_gb.transpose(1,2,0)
    new_img = remakeImage(new_img)
    new_img.save(savepath_gcam)
    
    # save_gradient_images(cam_gb, file_name_to_export + '_GGrad_Cam')
    # grayscale_cam_gb = convert_to_grayscale(cam_gb)
    # save_gradient_images(grayscale_cam_gb, file_name_to_export + '_GGrad_Cam_gray')
    print('Guided grad cam completed')

def grad(pretrained_model, prep_img, if_cuda, demog_label, target_class,
        savepath_grad, savepath_pos, savepath_neg):
    # Guided backprop
    GBP = GuidedBackprop(pretrained_model, if_cuda)
    # Get gradients
    guided_grads = GBP.generate_gradients(prep_img, target_class, demog_label)
    # Save colored gradients

    def save_img(grads, savepath):
        new_img = grads.transpose(1,2,0)
        new_img = remakeImage(new_img)
        new_img.save(savepath)

    save_img(guided_grads, savepath_grad)
    
    # Positive and negative saliency maps
    pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
    save_img(pos_sal, savepath_pos)
    save_img(neg_sal, savepath_neg)
    
    print('Guided backprop completed')

if __name__ == '__main__':
    if_cuda = True
    target_class = 3
    target_layer = 10
    demog_label = [3]
    demog = 'cm'

    savefolder = '/research/prip-gongsixu/codes/biasface/results/figures/neurIPS2020/visual/grad/gac/'+demog
    if os.path.isdir(savefolder) is False:
        os.makedirs(savefolder)

    savepath_grad = os.path.join(savefolder, 'grad.pdf')
    savepath_pos = os.path.join(savefolder, 'pos.pdf')
    savepath_neg = os.path.join(savefolder, 'neg.pdf')
    savepath_heat = os.path.join(savefolder, 'heat.pdf')
    savepath_heatimg = os.path.join(savefolder, 'heatimg.pdf')
    savepath_gcam = os.path.join(savefolder, 'gcam.pdf')
    
    path_pic = '/research/prip-gongsixu/codes/biasface/results/figures/neurIPS2020/visual/bf_sample.jpg'
    ori_img = Image.open(path_pic).convert('RGB')
    folder_model = '/research/prip-gongsixu/codes/biasface/results/models/attention/balance_attc_weight1.0_race/Save'
    path_model = os.path.join(folder_model,'model_epoch_25_final_0.943833.pth')

    pretrained_model = set_model(path_model, if_cuda)
    prep_img = load_image(path_pic, if_cuda)

    grad(pretrained_model, prep_img, if_cuda, demog_label, target_class,
        savepath_grad, savepath_pos, savepath_neg)
    grad_cam(pretrained_model, ori_img, prep_img, if_cuda, demog_label, 
        target_class, target_layer, savepath_heat, savepath_heatimg, savepath_gcam)