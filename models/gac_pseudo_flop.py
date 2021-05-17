# -*- coding: utf-8 -*-
"""
Created on 18-5-21 下午5:26

"""
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.utils.weight_norm as weight_norm
import torch.nn.functional as F

import pdb


__all__ = ['gac_pseudo50']


def conv3x3(ndemog, in_planes, out_planes, stride=1, adap=False, fuse_epoch=9):
    """3x3 convolution with padding"""
    return AdaConv2d(ndemog, in_planes, out_planes, 3, stride,
                     padding=1, adap=adap, fuse_epoch=fuse_epoch)
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                  padding=1, bias=False)

class IRBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, height=None, width=None, 
        downsample=None, use_se=True, use_att=False, use_spatial_att=False,
        ndemog=4, hard_att_channel=False, hard_att_spatial=False, lowresol_set={},
        adap=False, fuse_epoch=9):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(ndemog, inplanes, planes, stride, adap=adap, fuse_epoch=fuse_epoch)
        self.bn1 = nn.BatchNorm2d(planes)
        self.prelu1 = nn.PReLU(num_parameters=planes)
        self.conv2 = conv3x3(ndemog, planes, planes, adap=adap, fuse_epoch=fuse_epoch)
        self.bn2 = nn.BatchNorm2d(planes)
        self.prelu2 = nn.PReLU(num_parameters=planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        self.use_att = use_att
        if self.use_se:
            self.se = SEBlock(planes)
        if self.use_att:
            self.att = AttBlock(planes, height, width, ndemog, use_spatial_att,
                hard_att_channel, hard_att_spatial, lowresol_set)

    def forward(self, x):
        epoch = 0
        attc = None
        atts = None

        demog_label = torch.ones(x.size(0)).long()

        residual = x
        out = self.bn0(x)
        out = self.conv1(out, demog_label, epoch)
        out = self.bn1(out)
        out = self.prelu1(out)

        out = self.conv2(out, demog_label, epoch)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu2(out)

        if self.use_att:
            out,attc,atts = self.att(out, demog_label)

        return out

class AttBlock(nn.Module): # add more options, e.g, hard attention, low resolution attention
    def __init__(self, nchannel, height, width, ndemog=4, use_spatial_att=False,
        hard_att_channel=False, hard_att_spatial=False, lowersol_set={}):
        super(AttBlock, self).__init__()

        self.hard_att_channel = hard_att_channel
        self.hard_att_spatial = hard_att_spatial
        
        self.lowersol_mode = lowersol_set['mode']
        lowersol_rate = lowersol_set['rate']

        self.att_channel = nn.parameter.Parameter(torch.Tensor(1, 1, nchannel, 1, 1))
        nn.init.xavier_uniform_(self.att_channel)
        self.att_channel.data = self.att_channel.data.repeat(ndemog,1,1,1,1)

        self.use_spatial_att = use_spatial_att
        if use_spatial_att:
            self.height = int(height)
            self.width = int(width)
            self.att_spatial = nn.parameter.Parameter(torch.Tensor(ndemog, 1, 1, 
                int(height*lowersol_rate), int(width*lowersol_rate)))
            nn.init.xavier_uniform_(self.att_spatial)
        else:
            self.att_spatial = None

    def forward(self, x, demog_label):
        y = x
        demogs = list(set(demog_label.tolist()))

        if self.hard_att_channel:
            att_channel = torch.where(torch.sigmoid(self.att_channel) >= 0.5, 
                torch.ones_like(self.att_channel), torch.zeros_like(self.att_channel))
        else:
            att_channel = torch.sigmoid(self.att_channel)

        if self.use_spatial_att:
            if self.hard_att_spatial:
                att_spatial = torch.where(torch.sigmoid(self.att_spatial) >= 0.5, 
                    torch.ones_like(self.att_spatial), torch.zeros_like(self.att_spatial))
            else:
                att_spatial = torch.sigmoid(self.att_spatial)
            att_spatial = F.interpolate(att_spatial, size=(att_spatial.size(2), 
                self.height,self.width), mode=self.lowersol_mode)
        else:
            att_spatial = None

        if self.use_spatial_att:
            for demog in demogs:
                indices = torch.nonzero((demog_label==demog), as_tuple=False).squeeze()
                if indices.dim() == 0:
                    indices = indices.unsqueeze(0)
                y[indices,:,:,:] = x[indices,:,:,:] *\
                    att_channel.repeat(1, indices.size(0), 1, x.size(2), x.size(3))[demog,:,:,:,:] * \
                    att_spatial.repeat(1, indices.size(0), x.size(1), 1, 1)[demog,:,:,:,:]
        else:
            for demog in demogs:
                indices = torch.nonzero((demog_label==demog), as_tuple=False).squeeze()
                if indices.dim() == 0:
                    indices = indices.unsqueeze(0)
                y[indices,:,:,:] = x[indices,:,:,:] *\
                    att_channel.repeat(1, indices.size(0), 1, x.size(2), x.size(3))[demog,:,:,:,:]
        return y, att_channel, att_spatial

class AdaConv2d(nn.Module):
    def __init__(self, ndemog, ic, oc, ks, stride, padding=0, adap=True, fuse_epoch=9):
        super(AdaConv2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.fuse_epoch = fuse_epoch

        self.oc = oc
        self.adap = adap
        self.kernel_base = nn.Parameter(torch.Tensor(oc, ic, ks, ks))
        self.kernel_mask = nn.Parameter(torch.Tensor(1, ic, ks, ks))
        self.fuse_mark = nn.Parameter(torch.zeros(1))

        self.conv1 = nn.Conv2d(ic, oc, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(ic, oc, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.conv3 = nn.Conv2d(ic, oc, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.conv4 = nn.Conv2d(ic, oc, kernel_size=ks, stride=stride, padding=padding, bias=False)

        if adap:
            self.kernel_mask.data = self.kernel_mask.data.repeat(ndemog,1,1,1)

    def forward(self, x, demog_label, epoch):
        demogs = list(set(demog_label.tolist()))

        for i,demog in enumerate(demogs):
            # get indices
            indices = torch.nonzero((demog_label==demog), as_tuple=False).squeeze()
            if indices.dim() == 0:
                indices = indices.unsqueeze(0)
            
            # get mask
            if epoch >= self.fuse_epoch:
                if self.fuse_mark[0] == -1:
                    mask = self.kernel_mask[0,:,:,:].unsqueeze(0)
                else:
                    mask = self.kernel_mask[demog,:,:,:].unsqueeze(0)
            else:
                mask = self.kernel_mask[demog,:,:,:].unsqueeze(0)

            # get output
            if i == 0:
                temp = self.conv1(x[indices,:,:,:])
                size = [x.size(0)]
                for i in range(1,temp.dim()):
                    size.append(temp.size(i))
                output = torch.zeros(size)
                if x.is_cuda:
                    output = output.cuda()
            if demog == 0:
                output[indices,:,:,:] = self.conv1(x[indices,:,:,:])
            elif demog == 1:
                output[indices,:,:,:] = self.conv2(x[indices,:,:,:])
            elif demog == 2:
                output[indices,:,:,:] = self.conv3(x[indices,:,:,:])
            else:
                output[indices,:,:,:] = self.conv4(x[indices,:,:,:])
                

        return output

class ResNetFace(nn.Module):
    def __init__(self, block, layers, **kwargs,
        ):
        # use_se=False, use_spatial_att=False, ndemog=4, nclasses=2,
        self.inplanes = 64
        self.use_se = kwargs['use_se']
        self.use_spatial_att = kwargs['use_spatial_att']
        self.ndemog = kwargs['ndemog']
        self.hard_att_channel = kwargs['hard_att_channel']
        self.hard_att_spatial = kwargs['hard_att_spatial']
        self.lowresol_set = kwargs['lowresol_set']
        self.fuse_epoch = kwargs['fuse_epoch']

        super(ResNetFace, self).__init__()
        self.attinput = AttBlock(3, 112, 112, self.ndemog, self.use_spatial_att,
            self.hard_att_channel, self.hard_att_spatial, self.lowresol_set)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU(64)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.attconv1 = AttBlock(64, 56, 56, self.ndemog, self.use_spatial_att,
            self.hard_att_channel, self.hard_att_spatial, self.lowresol_set)
        
        self.layer1 = self._make_layer(block, 64, layers[0], height=56, width=56)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, height=28, width=28)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, height=14, width=14)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.bn4 = nn.BatchNorm2d(512)
        self.attbn4 = AttBlock(512, 7, 7, self.ndemog, self.use_spatial_att,
            self.hard_att_channel, self.hard_att_spatial, self.lowresol_set)
        
        self.dropout = nn.Dropout(p=0.4)
        self.fc5 = nn.Linear(512 * 7 * 7, 512)
        self.bn5 = nn.BatchNorm1d(512)

        # self.fc6 = nn.Linear(512,nclasses)
        # torch.nn.init.xavier_normal_(self.fc6.weight)

        for m in self.modules():
            if isinstance(m, AdaConv2d):
                nn.init.xavier_normal_(m.kernel_base)
                nn.init.xavier_normal_(m.kernel_mask)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, height=None, width=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, height, width,
            downsample, self.use_se, False, self.use_spatial_att,
            self.ndemog, self.hard_att_channel, self.hard_att_spatial, self.lowresol_set, 
            True, self.fuse_epoch))

        if height != None and width != None:
            use_att = True
        else:
            use_att = False

        self.inplanes = planes
        for i in range(1, blocks):
            if i == blocks-1:
                layers.append(block(self.inplanes, planes, 1, height, width,
                    None, self.use_se, use_att, self.use_spatial_att,
                    self.ndemog, self.hard_att_channel, self.hard_att_spatial, self.lowresol_set, 
                    True, self.fuse_epoch))
            else:
                layers.append(block(self.inplanes, planes, 1, height, width, 
                    None, self.use_se, False, self.use_spatial_att,
                    self.ndemog, self.hard_att_channel, self.hard_att_spatial, self.lowresol_set, 
                    True, self.fuse_epoch))

        return nn.Sequential(*layers)

    def forward(self, x):#, epoch):
        demog_label = torch.ones(x.size(0)).long()

        x,attc1,atts1 = self.attinput(x,demog_label)
        
        x = self.conv1(x) # 3 x 112 x 112
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x) # 64 x 56 x 56
        x,attc2,atts2 = self.attconv1(x,demog_label)

        x = self.layer1(x)#, 'epoch':epoch}) # 64 x 56 x 56

        x = self.layer2(x) # 128 x 28 x 28

        x = self.layer3(x) # 256 x 14 x 14

        x = self.layer4(x) # 512 x 7 x 7

        x = self.bn4(x)
        x6 = self.attbn4(x, demog_label)

        x = self.dropout(x)
        x = x.view(x.size(0), -1) # 1 x 25088
        x = self.fc5(x) # 1 x 512
        x = self.bn5(x)

        return x

def gac_pseudo18(use_se=False, **kwargs):
    model = ResNetFace(IRBlock, [2, 2, 2, 2], use_se=use_se, **kwargs)
    return model

def gac_pseudo34(use_se=False, **kwargs):
    model = ResNetFace(IRBlock, [3, 4, 6, 3], use_se=use_se, **kwargs)
    return model

def gac_pseudo50(use_se=False, **kwargs):
    model = ResNetFace(IRBlock, [3, 4, 14, 3], use_se=use_se, **kwargs)
    return model

def gac_pseudo100(use_se=False, **kwargs):
    model = ResNetFace(IRBlock, [3, 13, 30, 3], use_se=use_se, **kwargs)
    return model

def gac_pseudo152(use_se=False, **kwargs):
    model = ResNetFace(IRBlock, [3, 8, 36, 3], use_se=use_se, **kwargs)
    return model