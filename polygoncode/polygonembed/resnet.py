'''
COde rewritten based on https://github.com/ldeecke/mn-torch/blob/master/nn/resnet.py
'''


import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt
from torch.nn import BatchNorm1d

from polygonembed.atten import *

# from polygonembed.ops import ModeNorm

# resnet20 = lambda config: ResNet(BasicBlock, [3, 3, 3], config)
# resnet56 = lambda config: ResNet(BasicBlock, [9, 9, 9], config)
# resnet110 = lambda config: ResNet(BasicBlock, [18, 18, 18], config)


def get_agg_func(agg_func_type, out_channels, name = "resnet1d"):
    if agg_func_type == "mean":
        # global average pool
        return torch.mean
    elif agg_func_type == "min":
        # global min pool
        return torch.min
    elif agg_func_type == "max":
        # global max pool
        return torch.max
    elif agg_func_type.startswith("atten"):
        # agg_func_type: atten_whole_no_1
        atten_flag, att_type, bn, nat = agg_func_type.split("_")
        assert atten_flag == "atten"
        return AttentionSet(mode_dims = out_channels, 
                        att_reg = 0., 
                        att_tem = 1., 
                        att_type = att_type, 
                        bn = bn, 
                        nat= int(nat), 
                        name = name)


class ResNet1D(nn.Module):
    def __init__(self, block, num_layer_list, in_channels, out_channels, add_middle_pool = False, final_pool = "mean", padding_mode = 'circular', dropout_rate = 0.5):
        '''
        Args:
            block: BasicBlock() or BottleneckBlock()
            num_layer_list: [num_blocks0, num_blocks1, num_blocks2]
            inplanes: input number of channel
        '''
        super(ResNet1D, self).__init__()
        
        Norm = functools.partial(BatchNorm1d)

        self.num_layer_list = num_layer_list

        self.in_channels = in_channels
        # For simplicity, make outplanes dividable by block.expansion
        assert out_channels % block.expansion == 0
        self.out_channels = out_channels
        planes = int(out_channels / block.expansion)
        
        self.inplanes = out_channels

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode = padding_mode, bias=False)

        
        self.norm1 = Norm(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool1d(kernel_size = 2, stride=2, padding = 0)

        resnet_layers = []
        
        if len(self.num_layer_list) >= 1:
            # you have at least one number in self.num_layer_list
            layer1 = self._make_layer(block, planes, self.num_layer_list[0], Norm, padding_mode = padding_mode)
            resnet_layers.append(layer1)
            if add_middle_pool and len(self.num_layer_list) > 1:
                maxpool = nn.MaxPool1d(kernel_size = 2, stride=2, padding = 0)
                resnet_layers.append(maxpool)

        if len(self.num_layer_list) >= 2:
            # you have at least two numbers in self.num_layer_list
            for i in range(1, len(self.num_layer_list)):
                layerk = self._make_layer(block, planes, self.num_layer_list[i], Norm, stride=2, padding_mode = padding_mode)
                resnet_layers.append(layerk)
                if add_middle_pool and i < len(self.num_layer_list) - 1:
                    maxpool = nn.MaxPool1d(kernel_size = 2, stride=2, padding = 0)
                    resnet_layers.append(maxpool)

        self.resnet_layers = nn.Sequential(*resnet_layers)

        self.final_pool = final_pool

        self.final_pool_func = get_agg_func(agg_func_type = final_pool, 
                                            out_channels = out_channels, 
                                            name = "resnet1d")

        self.dropout = nn.Dropout(p=dropout_rate)
        
        # if len(self.num_layer_list) == 3:
        #     # you have three numbers in self.num_layer_list
        #     self.layer3 = self._make_layer(block, planes, self.num_layer_list[2], Norm, stride=2, padding_mode = padding_mode)
        
        # self.avgpool = nn.AvgPool1d(8, stride=1)
        # self.fc = nn.Linear(64 * block.expansion, num_classes)

        self._init_weights()

    # def finalPool1d_func(self, final_pool, out_channels, name = "resnet1d"):
    #     if final_pool == "mean":
    #         # global average pool
    #         return torch.mean
    #     elif final_pool == "min":
    #         # global min pool
    #         return torch.min
    #     elif final_pool == "max":
    #         # global max pool
    #         return torch.max
    #     elif final_pool.startswith("atten"):
    #         # final_pool: atten_whole_no_1
    #         atten_flag, att_type, bn, nat = final_pool.split("_")
    #         assert atten_flag == "atten"
    #         return AttentionSet(mode_dims = out_channels, 
    #                         att_reg = 0., 
    #                         att_tem = 1., 
    #                         att_type = att_type, 
    #                         bn = bn, 
    #                         nat= int(nat), 
    #                         name = name)

    def finalPool1d(self, x, final_pool = "mean"):
        '''
        Args:
            x: shape (batch_size, out_channels, (seq_len+k-2)/2^k )
        Return:

        '''
        if final_pool == "mean":
            # global average pool
            # x: shape (batch_size, out_channels)
            x = self.final_pool_func(x, dim = -1, keepdim = False)
        elif final_pool == "min":
            # global min pool
            # x: shape (batch_size, out_channels)
            x, indice = self.final_pool_func(x, dim = -1, keepdim = False)
        elif final_pool == "max":
            # global max pool
            # x: shape (batch_size, out_channels)
            x, indice = self.final_pool_func(x, dim = -1, keepdim = False)
        elif final_pool.startswith("atten"):
            # attenion based aggregation
            # x: shape (batch_size, out_channels)
            x = self.final_pool_func(x)
        return x

    def forward(self, x):
        '''
        Args:
            x: shape (batch_size, in_channels, seq_len)
        Return:
            x: shape (batch_size, out_channels)
        '''
        # x: shape (batch_size, out_channels, seq_len)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        # print("conv1:", x.shape)

        # x: shape (batch_size, out_channels, seq_len/2)
        x = self.maxpool(x)

        # x: shape (batch_size, out_channels, (seq_len+k-2)/2^k )
        x = self.resnet_layers(x)

        # if len(self.num_layer_list) >= 1:
        #     # After 1st block: shape (batch_size, out_channels, seq_len/2)
        #     # x: shape (batch_size, out_channels, seq_len/2)
        #     x = self.layer1(x)
        #     # print("layer1:", x.shape)

        # if len(self.num_layer_list) >= 2:
        #     # After 1st block: shape (batch_size, out_channels, (seq_len+2)/4 )
        #     # x: shape (batch_size, out_channels, (seq_len+2)/4 )
        #     x = self.layer2(x)
        #     # print("layer2:", x.shape)

        # if len(self.num_layer_list) == 3:
        #     # After 1st block: shape (batch_size, out_channels, (seq_len+6)/8 )
        #     # x: shape (batch_size, out_channels, (seq_len+6)/8 )
        #     x = self.layer3(x)
        #     # print("layer3:", x.shape)



        # global pool
        # x: shape (batch_size, out_channels)
        x = self.finalPool1d(x, self.final_pool)
        # x = torch.mean(x, dim = -1, keepdim = False)
        # print("avgpool:", x.shape)


        x = self.dropout(x)

        # x: shape (batch_size, 64*expansion, (seq_len-25)/4 )
        # x = self.avgpool(x)
        # x: shape (batch_size, 64*expansion * (seq_len-25)/4 )
        # x = x.view(x.size(0), -1)
        # x: shape (batch_size, 64*expansion * (seq_len-25)/4 )
        # x = self.fc(x)
        # print("output:", x.shape)

        return x


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, sqrt(2./n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, ModeNorm):
            #     m.alpha.data.fill_(1)
            #     m.beta.data.zero_()


    def _make_layer(self, block, planes, blocks, norm, stride=1, padding_mode = 'circular'):
        downsample = None
        if (stride != 1) or (self.inplanes != planes * block.expansion):
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, norm, stride, downsample, padding_mode))
        # self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm, padding_mode = padding_mode))

        return nn.Sequential(*layers)


class ResNet1D3(nn.Module):
    def __init__(self, block, num_layer_list, in_channels, out_channels, padding_mode = 'circular'):
        '''
        Args:
            block: BasicBlock() or BottleneckBlock()
            num_layer_list: [num_blocks0, num_blocks1, num_blocks2]
            inplanes: input number of channel
        '''
        super(ResNet1D3, self).__init__()
        
        Norm = functools.partial(BatchNorm1d)

        self.num_layer_list = num_layer_list

        self.in_channels = in_channels
        # For simplicity, make outplanes dividable by block.expansion
        assert out_channels % block.expansion == 0
        self.out_channels = out_channels
        planes = int(out_channels / block.expansion)
        
        self.inplanes = out_channels

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode = padding_mode, bias=False)

        
        self.norm1 = Norm(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool1d(kernel_size = 2, stride=2, padding = 0)
        
        if len(self.num_layer_list) >= 1:
            # you have at least one number in self.num_layer_list
            self.layer1 = self._make_layer(block, planes, self.num_layer_list[0], Norm, padding_mode = padding_mode)
        
        if len(self.num_layer_list) >= 2:
            # you have at least two numbers in self.num_layer_list
            self.layer2 = self._make_layer(block, planes, self.num_layer_list[1], Norm, stride=2, padding_mode = padding_mode)
        
        if len(self.num_layer_list) == 3:
            # you have three numbers in self.num_layer_list
            self.layer3 = self._make_layer(block, planes, self.num_layer_list[2], Norm, stride=2, padding_mode = padding_mode)
        
        # self.avgpool = nn.AvgPool1d(8, stride=1)
        # self.fc = nn.Linear(64 * block.expansion, num_classes)

        self._init_weights()


    def forward(self, x):
        '''
        Args:
            x: shape (batch_size, in_channels, seq_len)
        Return:
            x: shape (batch_size, out_channels)
        '''
        # x: shape (batch_size, out_channels, seq_len)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        # print("conv1:", x.shape)

        # x: shape (batch_size, out_channels, seq_len/2)
        x = self.maxpool(x)

        if len(self.num_layer_list) >= 1:
            # After 1st block: shape (batch_size, out_channels, seq_len/2)
            # x: shape (batch_size, out_channels, seq_len/2)
            x = self.layer1(x)
            # print("layer1:", x.shape)

        if len(self.num_layer_list) >= 2:
            # After 1st block: shape (batch_size, out_channels, (seq_len+2)/4 )
            # x: shape (batch_size, out_channels, (seq_len+2)/4 )
            x = self.layer2(x)
            # print("layer2:", x.shape)

        if len(self.num_layer_list) == 3:
            # After 1st block: shape (batch_size, out_channels, (seq_len+6)/8 )
            # x: shape (batch_size, out_channels, (seq_len+6)/8 )
            x = self.layer3(x)
            # print("layer3:", x.shape)

        # global average pool
        # x: shape (batch_size, out_channels)
        x = torch.mean(x, dim = -1, keepdim = False)
        # print("avgpool:", x.shape)

        # x: shape (batch_size, 64*expansion, (seq_len-25)/4 )
        # x = self.avgpool(x)
        # x: shape (batch_size, 64*expansion * (seq_len-25)/4 )
        # x = x.view(x.size(0), -1)
        # x: shape (batch_size, 64*expansion * (seq_len-25)/4 )
        # x = self.fc(x)
        # print("output:", x.shape)

        return x


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, sqrt(2./n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, ModeNorm):
            #     m.alpha.data.fill_(1)
            #     m.beta.data.zero_()


    def _make_layer(self, block, planes, blocks, norm, stride=1, padding_mode = 'circular'):
        downsample = None
        if (stride != 1) or (self.inplanes != planes * block.expansion):
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, norm, stride, downsample, padding_mode))
        # self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm, padding_mode = padding_mode))

        return nn.Sequential(*layers)


class ResNet1DLucas(nn.Module):
    def __init__(self, block, layers, inplanes, num_classes, padding_mode = 'circular'):
        '''
        Args:
            layers: [num_blocks0, num_blocks1, num_blocks2]
        '''
        super(ResNet1DLucas, self).__init__()
        # self.mn = config.mn

        # if config.mn == "full":
        #     Norm = functools.partial(ModeNorm, momentum=config.momentum, n_components=config.num_components)
        # elif config.mn == "init":
        #     InitNorm = functools.partial(ModeNorm, momentum=config.momentum, n_components=config.num_components)
        #     Norm = functools.partial(BatchNorm1d, momentum=config.momentum)
        Norm = functools.partial(BatchNorm1d)

        self.inplanes = 16
        self.conv1 = nn.Conv1d(inplanes, 16, kernel_size=3, stride=1, padding=1, padding_mode = padding_mode, bias=False)

        # self.norm1 = InitNorm(16) if config.mn == "init" else Norm(16)
        self.norm1 = Norm(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0], Norm, padding_mode = padding_mode)
        self.layer2 = self._make_layer(block, 32, layers[1], Norm, stride=2, padding_mode = padding_mode)
        self.layer3 = self._make_layer(block, 64, layers[2], Norm, stride=2, padding_mode = padding_mode)
        # self.avgpool = nn.AvgPool1d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        self._init_weights()


    def forward(self, x):
        '''
        Args:
            x: shape (batch_size, 3, seq_len)
        '''
        # x: shape (batch_size, 16, seq_len)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        # After 1st block: shape (batch_size, 16*expansion, seq_len)
        # x: shape (batch_size, 16*expansion, seq_len)
        x = self.layer1(x)

        # After 1st block: shape (batch_size, 32*expansion, (seq_len+1)/2 )
        # x: shape (batch_size, 32*expansion, (seq_len+1)/2 )
        x = self.layer2(x)

        # After 1st block: shape (batch_size, 64*expansion, (seq_len+3)/4 )
        # x: shape (batch_size, 64*expansion, (seq_len+3)/4 )
        x = self.layer3(x)

        # global average pool
        # x: shape (batch_size, 64*expansion)
        x = torch.mean(x, dim = -1, keepdim = False)

        # x: shape (batch_size, 64*expansion, (seq_len-25)/4 )
        # x = self.avgpool(x)
        # x: shape (batch_size, 64*expansion * (seq_len-25)/4 )
        # x = x.view(x.size(0), -1)
        # x: shape (batch_size, 64*expansion * (seq_len-25)/4 )
        x = self.fc(x)

        return x


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, sqrt(2./n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, ModeNorm):
            #     m.alpha.data.fill_(1)
            #     m.beta.data.zero_()


    def _make_layer(self, block, planes, blocks, norm, stride=1, padding_mode = 'circular'):
        downsample = None
        if (stride != 1) or (self.inplanes != planes * block.expansion):
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, norm, stride, downsample, padding_mode))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm, padding_mode = padding_mode))

        return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, norm, stride=1, downsample=None, padding_mode = 'circular'):
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv3(inplanes, planes, stride, padding_mode = padding_mode)
        self.norm1 = norm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self._conv3(planes, planes, padding_mode = padding_mode)
        self.norm2 = norm(planes)
        self.downsample = downsample
        self.stride = stride
        self.padding_mode = padding_mode


    def forward(self, x):
        '''
        Args:
            x: shape (batch_size, in_planes, seq_len)
        return:
            out: shape (batch_size, planes, (seq_len-1)/stride + 1 )
        '''
        residual = x

        # out: shape (batch_size, planes, (seq_len-1)/stride + 1 )
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        # out: shape (batch_size, planes, (seq_len-1)/stride + 1 )
        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            # residual: shape (batch_size, planes, (seq_len-1)/stride + 1 )
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


    def _conv3(self, in_planes, out_planes, stride=1, padding_mode = 'circular'):
        '''3x3 convolution with padding'''
        return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, padding_mode = padding_mode, bias=False)


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, norm, stride=1, downsample=None, padding_mode = 'circular', ):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.norm1 = norm(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, padding_mode = padding_mode, bias=False)
        self.norm2 = norm(planes)
        self.conv3 = nn.Conv1d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.norm3 = norm(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                norm(self.expansion*planes)
            )


    def forward(self, x):
        '''
        Args:
            x: shape (batch_size, in_planes, seq_len)
        return:
            out: shape (batch_size, expansion*planes, (seq_len-1)/stride + 1 )
        '''
        # out: shape (batch_size, planes, seq_len)
        out = F.relu(self.norm1(self.conv1(x)))
        # out: shape (batch_size, planes, (seq_len-1)/stride + 1 )
        out = F.relu(self.norm2(self.conv2(out)))
        # out: shape (batch_size, expansion*planes, (seq_len-1)/stride + 1 )
        out = self.norm3(self.conv3(out))
        # self.shortcut(x): shape (batch_size, expansion*planes, (seq_len-1)/stride + 1 )
        # out: shape (batch_size, expansion*planes, (seq_len-1)/stride + 1 )
        out += self.shortcut(x)
        out = F.relu(out)
        return out
