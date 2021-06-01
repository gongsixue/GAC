# -*- coding: utf-8 -*-
"""
Created on 18-5-21 下午5:26

@author: ronghuaiyang
"""
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.utils.weight_norm as weight_norm
import torch.nn.functional as F

import pdb


__all__ = ['subnet_face50']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class IRBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, height=None, width=None, 
        downsample=None, use_se=True, use_att=False, use_spatial_att=False,
        ndemog=4, hard_att_channel=False, hard_att_spatial=False, lowresol_set={}):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.prelu1 = nn.PReLU(num_parameters=planes)
        self.conv2 = conv3x3(planes, planes, stride)
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

    def forward(self, x_dict):
        x = x_dict['x']
        demog_label = x_dict['demog_label']
        attc = None
        atts = None

        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu2(out)

        if self.use_att:
            out,attc,atts = self.att(out, demog_label)

        return {'x':out, 'demog_label':demog_label, 'attc':attc, 'atts':atts}

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.PReLU(),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class AttBlock(nn.Module):
    def __init__(self, nchannel, height, width, ndemogs=4, use_spatial_att=False,
        hard_att_channel=False, hard_att_spatial=False, lowersol_set={}):
        super(AttBlock, self).__init__()

        self.hard_att_channel = hard_att_channel
        self.hard_att_spatial = hard_att_spatial
        
        self.lowersol_mode = lowersol_set['mode']
        lowersol_rate = lowersol_set['rate']

        self.att_channel = nn.parameter.Parameter(torch.Tensor(ndemogs, 1, nchannel, 1, 1))
        nn.init.xavier_uniform_(self.att_channel)

        self.use_spatial_att = use_spatial_att
        if use_spatial_att:
            self.att_spatial = nn.parameter.Parameter(torch.Tensor(ndemogs, 1, 1, 
                height, width))
            nn.init.xavier_uniform_(self.att_spatial)
        else:
            self.att_spatial = None

    def forward(self, x, demog_label):
        y = x
        demogs = list(set(demog_label.tolist()))

        att_channel = self.att_channel
        if self.use_spatial_att:
            att_spatial = self.att_spatial
        else:
            att_spatial = None

        if self.use_spatial_att:
            for demog in demogs:
                indices = (demog_label==demog).nonzero().squeeze()
                indices = indices.unsqueeze(0)
                y[indices,:,:,:].sq = x[indices,:,:,:] *\
                    att_channel.repeat(1, indices.size(0), 1, x.size(2), x.size(3))[demog,:,:,:,:] * \
                    att_spatial.repeat(1, indices.size(0), x.size(1), 1, 1)[demog,:,:,:,:]
        else:
            for demog in demogs:
                indices = (demog_label==demog).nonzero().squeeze()
                indices = indices.unsqueeze(0)
                y[indices,:,:,:].sq = x[indices,:,:,:] *\
                    att_channel.repeat(1, indices.size(0), 1, x.size(2), x.size(3))[demog,:,:,:,:]
        return y, att_channel, att_spatial

class AttBlock_new(nn.Module): # add more options, e.g, hard attention, low resolution attention
    def __init__(self, nchannel, height, width, ndemogs=4, use_spatial_att=False,
        hard_att_channel=False, hard_att_spatial=False, lowersol_set={}):
        super(AttBlock_new, self).__init__()

        self.hard_att_channel = hard_att_channel
        self.hard_att_spatial = hard_att_spatial
        
        self.lowersol_mode = lowersol_set['mode']
        lowersol_rate = lowersol_set['rate']

        self.att_channel = nn.parameter.Parameter(torch.Tensor(ndemogs, 1, nchannel, 1, 1))
        nn.init.xavier_uniform_(self.att_channel)

        self.use_spatial_att = use_spatial_att
        if use_spatial_att:
            self.height = int(height)
            self.width = int(width)
            self.att_spatial = nn.parameter.Parameter(torch.Tensor(ndemogs, 1, 1, 
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
                indices = (demog_label==demog).nonzero().squeeze()
                indices = indices.unsqueeze(0)
                y[indices,:,:,:].sq = x[indices,:,:,:] *\
                    att_channel.repeat(1, indices.size(0), 1, x.size(2), x.size(3))[demog,:,:,:,:] * \
                    att_spatial.repeat(1, indices.size(0), x.size(1), 1, 1)[demog,:,:,:,:]
        else:
            for demog in demogs:
                indices = (demog_label==demog).nonzero().squeeze()
                indices = indices.unsqueeze(0)
                y[indices,:,:,:].sq = x[indices,:,:,:] *\
                    att_channel.repeat(1, indices.size(0), 1, x.size(2), x.size(3))[demog,:,:,:,:]
        return y, att_channel, att_spatial

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
            if isinstance(m, nn.Conv2d):
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
            self.ndemog, self.hard_att_channel, self.hard_att_spatial, self.lowresol_set))

        if height != None and width != None:
            use_att = True
        else:
            use_att = False

        self.inplanes = planes
        for i in range(1, blocks):
            if i == blocks-1:
                layers.append(block(self.inplanes, planes, 1, height, width,
                    None, self.use_se, use_att, self.use_spatial_att,
                    self.ndemog, self.hard_att_channel, self.hard_att_spatial, self.lowresol_set))
            else:
                layers.append(block(self.inplanes, planes, 1, height, width, 
                    None, self.use_se, False, self.use_spatial_att,
                    self.ndemog, self.hard_att_channel, self.hard_att_spatial, self.lowresol_set))

        return nn.Sequential(*layers)

    def forward(self, x, demog_label):
        x,attc1,atts1 = self.attinput(x,demog_label)
        
        x = self.conv1(x) # 3 x 112 x 112
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x) # 64 x 56 x 56
        x,attc2,atts2 = self.attconv1(x,demog_label)

        x_dict = self.layer1({'x':x, 'demog_label':demog_label}) # 64 x 56 x 56
        attc3 = x_dict['attc']
        atts3 = x_dict['atts']

        x_dict = self.layer2(x_dict) # 128 x 28 x 28
        attc4 = x_dict['attc']
        atts4 = x_dict['atts']

        x_dict = self.layer3(x_dict) # 256 x 14 x 14
        attc5 = x_dict['attc']
        atts5 = x_dict['atts']

        x_dict = self.layer4(x_dict) # 512 x 7 x 7

        x = x_dict['x']
        x = self.bn4(x)
        x,attc6,atts6 = self.attbn4(x, demog_label)

        x = self.dropout(x)
        x = x.view(x.size(0), -1) # 1 x 25088
        x = self.fc5(x) # 1 x 512
        x = self.bn5(x)

        return x

def attdemog_face18(use_se=False, **kwargs):
    model = ResNetFace(IRBlock, [2, 2, 2, 2], use_se=use_se, **kwargs)
    return model

def attdemog_face34(use_se=False, **kwargs):
    model = ResNetFace(IRBlock, [3, 4, 6, 3], use_se=use_se, **kwargs)
    return model

def subnet_face50(use_se=False, **kwargs):
    model = ResNetFace(IRBlock, [3, 4, 14, 3], use_se=use_se, **kwargs)
    return model

def attdemog_face100(use_se=False, **kwargs):
    model = ResNetFace(IRBlock, [3, 13, 30, 3], use_se=use_se, **kwargs)
    return model

def attdemog_face152(use_se=False, **kwargs):
    model = ResNetFace(IRBlock, [3, 8, 36, 3], use_se=use_se, **kwargs)
    return model