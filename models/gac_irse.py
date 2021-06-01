from collections import namedtuple
import math

import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
import pdb

__all__ = ['gac_IR_SE_50', 'gac_IR_SE_101', 'gac_IR_SE_152']

##################################  Original Arcface Model #############################################################

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0 ,bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0 ,bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class AttBlock(nn.Module): # add more options, e.g, hard attention, low resolution attention
    def __init__(self, nchannel, height, width, ndemog=4, use_spatial_att=False,
        hard_att_channel=False, hard_att_spatial=False, lowresol_set={}):
        super(AttBlock, self).__init__()
        self.ndemog = ndemog

        self.hard_att_channel = hard_att_channel
        self.hard_att_spatial = hard_att_spatial
        
        self.lowersol_mode = lowresol_set['mode']
        lowersol_rate = lowresol_set['rate']

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
        # demogs = list(set(demog_label.tolist()))
        demogs = list(range(self.ndemog))

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
        self.ic = ic
        self.ks = ks
        self.ndemog = ndemog
        self.adap = adap
        # self.kernel_adap = nn.Parameter(torch.Tensor(ndemog, oc, ic, ks, ks))
        self.kernel_base = nn.Parameter(torch.Tensor(oc, ic, ks, ks))
        # self.kernel_net = nn.Linear(ndemog, oc*ic*ks*ks)
        self.kernel_mask = nn.Parameter(torch.Tensor(1, ic, ks, ks))
        # self.fuse_mark = nn.Parameter(torch.zeros(1))
        self.fuse_mark = nn.Parameter(torch.Tensor(1))

        # self.conv = nn.Conv2d(ic, oc, kernel_size=3, stride=stride,
        #              padding=1, bias=False)
        # self.conv.weight = self.kernel_base

        if adap:
            self.kernel_mask.data = self.kernel_mask.data.repeat(ndemog,1,1,1)

    def forward(self, x, demog_label, epoch):
        demogs = list(range(self.ndemog))

        if self.adap:
            for i,demog in enumerate(demogs):
                # get indices
                indices = torch.nonzero((demog_label==demog), as_tuple=False).squeeze()
                if indices.dim() == 0:
                    indices = indices.unsqueeze(0)

                # k_input = F.one_hot(demog_label, num_classes=self.ndemog)
                
                # get mask
                if epoch >= self.fuse_epoch:
                    if self.fuse_mark[0] == -1:
                        mask = self.kernel_mask[0,:,:,:].unsqueeze(0)
                        # kernel_mask = self.kernel_adap[0,:,:,:,:]
                    else:
                        mask = self.kernel_mask[demog,:,:,:].unsqueeze(0)
                        # kernel_mask = self.kernel_adap[demog,:,:,:,:]
                else:
                    mask = self.kernel_mask[demog,:,:,:].unsqueeze(0)
                    # kernel_mask = self.kernel_adap[demog,:,:,:,:]
                # if epoch >= self.fuse_epoch:
                #     if self.fuse_mark[0] == -1:
                #         kernel_mask = self.kernel_adap[0,:,:,:,:]
                #         # k_input = F.one_hot([0], num_classes=self.ndemog)
                #     else:
                #         kernel_mask = self.kernel_adap[demog,:,:,:,:]
                #         # k_input = F.one_hot(demog_label, num_classes=self.ndemog)
                # else:
                #     kernel_mask = self.kernel_adap[demog,:,:,:,:]
                #     # k_input = F.one_hot(demog_label, num_classes=self.ndemog)
                # # kernel_mask = self.kernel_net(k_input)

                # get output
                if i == 0:
                    temp = F.conv2d(x[indices,:,:,:], self.kernel_base*mask.repeat(self.oc,1,1,1),
                        stride=self.stride, padding=self.padding)
                    # temp = self.conv(x[indices,:,:,:])
                    # initialize output
                    size = [x.size(0)]
                    for i in range(1,temp.dim()):
                        size.append(temp.size(i))
                    output = torch.zeros(size)
                    if x.is_cuda:
                        output = output.cuda()                    
                    output[indices,:,:,:] = temp
                else:
                    output[indices,:,:,:] = F.conv2d(x[indices,:,:,:], 
                        self.kernel_base*mask.repeat(self.oc,1,1,1),
                        stride=self.stride, padding=self.padding)
                    # output[indices,:,:,:] = self.conv(x[indices,:,:,:])
        else:
            output = F.conv2d(x, self.kernel_base, stride=self.stride, padding=self.padding)
            # output = self.conv(x)
            # k_input = F.one_hot(torch.tensor([0]), num_classes=self.ndemog)
            # kernel_mask = self.kernel_net(k_input.float())
            # kernel_mask = kernel_mask.view(self.oc, self.ic, self.ks, self.ks)
            # if x.is_cuda:
            #     kernel_mask = kernel_mask.cuda()

        return output

class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride ,bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1 ,bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1 ,bias=False), BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride, ndemog, adap, fuse_epoch,
        use_att, height, width, use_spatial_att, hard_att_channel, hard_att_spatial, lowresol_set):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride ,bias=False), 
                BatchNorm2d(depth))
        # self.res_layer = Sequential(
        #     BatchNorm2d(in_channel),
        #     AdaConv2d(ndemog, in_channel, depth, 3, 1,1 ,adap, fuse_epoch),
        #     PReLU(depth),
        #     AdaConv2d(ndemog, depth, depth, 3, stride, 1 ,adap, fuse_epoch),
        #     BatchNorm2d(depth),
        #     SEModule(depth,16)
        #     )
        self.bn0 = BatchNorm2d(in_channel)
        self.conv1 = AdaConv2d(ndemog, in_channel, depth, 3, 1,1 ,adap, fuse_epoch)
        self.prelu = PReLU(depth)
        self.conv2 = AdaConv2d(ndemog, depth, depth, 3, stride, 1 ,adap, fuse_epoch)
        self.bn1 = BatchNorm2d(depth)
        self.use_se = SEModule(depth,16)

        self.use_att = use_att
        if self.use_att:
            self.att = AttBlock(depth, height, width, ndemog, use_spatial_att,
                hard_att_channel, hard_att_spatial, lowresol_set)

    def forward(self,inputs):
        x = inputs['x']
        demog_label = inputs['demog_label']
        epoch = inputs['epoch']
        shortcut = self.shortcut_layer(x)
        x = self.bn0(x)
        x = self.conv1(x, demog_label, epoch)
        x = self.prelu(x)
        x = self.conv2(x, demog_label, epoch)
        x = self.bn1(x)
        x = self.use_se(x)
        x = x + shortcut

        attc = None
        atts = None

        if self.use_att:
            x,attc,atts = self.att(x, demog_label)
        return {'x':x, 'demog_label':demog_label, 'epoch':epoch, 'attc':attc, 'atts':atts}

class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride', 
    'ndemog', 'adap', 'fuse_epoch',
    'use_att', 'height', 'width', 'use_spatial_att', 'hard_att_channel', 'hard_att_spatial', 'lowresol_set'])):
    '''A named tuple describing a ResNet block.'''
    
def get_block(in_channel, depth, num_units, ndemog, adap, fuse_epoch, 
    use_att, height, width, use_spatial_att, hard_att_channel, hard_att_spatial, lowresol_set):
  return [Bottleneck(in_channel, depth, 2, ndemog, adap, fuse_epoch, 
    use_att, height, width, use_spatial_att, hard_att_channel, hard_att_spatial, lowresol_set)] + \
    [Bottleneck(depth, depth, 1, ndemog, adap, fuse_epoch,
        use_att, height, width, use_spatial_att, hard_att_channel, hard_att_spatial, lowresol_set) \
        for i in range(num_units-1)]

def get_blocks(num_layers, ndemog, fuse_epoch, att_dict):
    use_spatial_att = att_dict['use_spatial_att']
    hard_att_channel = att_dict['hard_att_channel']
    hard_att_spatial = att_dict['hard_att_spatial']
    lowresol_set = att_dict['lowresol_set']
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units = 3, 
                ndemog=ndemog, adap=True, fuse_epoch=fuse_epoch,
                use_att=True, height=56, width=56, use_spatial_att=use_spatial_att, 
                hard_att_channel=hard_att_channel, hard_att_spatial=hard_att_spatial, 
                lowresol_set=lowresol_set),
            get_block(in_channel=64, depth=128, num_units=4, 
                ndemog=ndemog, adap=True, fuse_epoch=fuse_epoch,
                use_att=True, height=28, width=28, use_spatial_att=use_spatial_att, 
                hard_att_channel=hard_att_channel, hard_att_spatial=hard_att_spatial, 
                lowresol_set=lowresol_set),
            get_block(in_channel=128, depth=256, num_units=14, 
                ndemog=ndemog, adap=True, fuse_epoch=fuse_epoch,
                use_att=True, height=14, width=14, use_spatial_att=use_spatial_att, 
                hard_att_channel=hard_att_channel, hard_att_spatial=hard_att_spatial, 
                lowresol_set=lowresol_set),
            get_block(in_channel=256, depth=512, num_units=3, 
                ndemog=ndemog, adap=True, fuse_epoch=fuse_epoch,
                use_att=True, height=7, width=7, use_spatial_att=use_spatial_att, 
                hard_att_channel=hard_att_channel, hard_att_spatial=hard_att_spatial, 
                lowresol_set=lowresol_set)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3, 
                ndemog=ndemog, adap=True, fuse_epoch=fuse_epoch,
                use_att=True, height=56, width=56, use_spatial_att=use_spatial_att, 
                hard_att_channel=hard_att_channel, hard_att_spatial=hard_att_spatial, 
                lowresol_set=lowresol_set),
            get_block(in_channel=64, depth=128, num_units=13, 
                ndemog=ndemog, adap=True, fuse_epoch=fuse_epoch,
                use_att=True, height=28, width=28, use_spatial_att=use_spatial_att, 
                hard_att_channel=hard_att_channel, hard_att_spatial=hard_att_spatial, 
                lowresol_set=lowresol_set),
            get_block(in_channel=128, depth=256, num_units=30, 
                ndemog=ndemog, adap=True, fuse_epoch=fuse_epoch,
                use_att=True, height=14, width=14, use_spatial_att=use_spatial_att, 
                hard_att_channel=hard_att_channel, hard_att_spatial=hard_att_spatial, 
                lowresol_set=lowresol_set),
            get_block(in_channel=256, depth=512, num_units=3, 
                ndemog=ndemog, adap=True, fuse_epoch=fuse_epoch,
                use_att=True, height=7, width=7, use_spatial_att=use_spatial_att, 
                hard_att_channel=hard_att_channel, hard_att_spatial=hard_att_spatial, 
                lowresol_set=lowresol_set)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3, 
                ndemog=ndemog, adap=True, fuse_epoch=fuse_epoch,
                use_att=True, height=56, width=56, use_spatial_att=use_spatial_att, 
                hard_att_channel=hard_att_channel, hard_att_spatial=hard_att_spatial, 
                lowresol_set=lowresol_set),
            get_block(in_channel=64, depth=128, num_units=8, 
                ndemog=ndemog, adap=True, fuse_epoch=fuse_epoch,
                use_att=True, height=28, width=28, use_spatial_att=use_spatial_att, 
                hard_att_channel=hard_att_channel, hard_att_spatial=hard_att_spatial, 
                lowresol_set=lowresol_set),
            get_block(in_channel=128, depth=256, num_units=36, 
                ndemog=ndemog, adap=True, fuse_epoch=fuse_epoch,
                use_att=True, height=14, width=14, use_spatial_att=use_spatial_att, 
                hard_att_channel=hard_att_channel, hard_att_spatial=hard_att_spatial, 
                lowresol_set=lowresol_set),
            get_block(in_channel=256, depth=512, num_units=3, 
                ndemog=ndemog, adap=True, fuse_epoch=fuse_epoch,
                use_att=True, height=7, width=7, use_spatial_att=use_spatial_att, 
                hard_att_channel=hard_att_channel, hard_att_spatial=hard_att_spatial, 
                lowresol_set=lowresol_set)
        ]
    return blocks

class Backbone(Module):
    def __init__(self, num_layers, mode, **kwargs):
        super(Backbone, self).__init__()
        # assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"

        # self.use_se = kwargs['use_se']
        use_spatial_att = kwargs['use_spatial_att']
        self.ndemog = kwargs['ndemog']
        hard_att_channel = kwargs['hard_att_channel']
        hard_att_spatial = kwargs['hard_att_spatial']
        lowresol_set = kwargs['lowresol_set']
        self.fuse_epoch = kwargs['fuse_epoch']
        
        blocks = get_blocks(num_layers, self.ndemog, self.fuse_epoch,
            {'use_spatial_att':use_spatial_att, 'hard_att_channel':hard_att_channel, \
            'hard_att_spatial':hard_att_spatial, 'lowresol_set':lowresol_set})
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        # if input_size[0] == 112:
        self.output_layer = Sequential(BatchNorm2d(512),
                                       Dropout(0.4),
                                       Flatten(),
                                       Linear(512 * 7 * 7, 512),
                                       BatchNorm1d(512, affine=False))
        # else:
        #     self.output_layer = Sequential(BatchNorm2d(512),
        #                                    Dropout(0.4),
        #                                    Flatten(),
        #                                    Linear(512 * 14 * 14, 512),
        #                                    BatchNorm1d(512, affine=False))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride,
                                bottleneck.ndemog, bottleneck.adap, bottleneck.fuse_epoch,
                                bottleneck.use_att, bottleneck.height, bottleneck.width,
                                bottleneck.use_spatial_att, bottleneck.hard_att_channel,
                                bottleneck.hard_att_spatial, bottleneck.lowresol_set))
        self.body = Sequential(*modules)

        self._initialize_weights()

    def forward(self, inputs, epoch):
        x = inputs[0]
        demog_label = inputs[1]
        x = self.input_layer(x)
        x_dict = self.body({'x':x, 'demog_label':demog_label, 'epoch':epoch})
        x = x_dict['x']
        attc = [x_dict['attc']]
        atts = [x_dict['atts']]
        conv_out = x.view(x.shape[0], -1)
        x = self.output_layer(x)

        return x, {'attc':attc,'atts':atts}

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, AdaConv2d):
                nn.init.xavier_normal_(m.kernel_base)
                nn.init.xavier_normal_(m.kernel_mask)
                nn.init.constant_(m.fuse_mark.data, val=-1)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()


def gac_IR_SE_50(**kwargs):
    """Constructs a ir_se-50 model.
    """
    model = Backbone(50, 'ir_se', **kwargs)

    return model


def gac_IR_SE_101(**kwargs):
    """Constructs a ir_se-101 model.
    """
    model = Backbone(100, 'ir_se', **kwargs)

    return model


def gac_IR_SE_152(**kwargs):
    """Constructs a ir_se-152 model.
    """
    model = Backbone(152, 'ir_se', **kwargs)

    return model