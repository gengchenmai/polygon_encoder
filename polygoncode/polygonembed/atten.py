import numpy as np
import random
import pickle
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class AttentionSet(nn.Module):
    def __init__(self, mode_dims, att_reg=0., att_tem=1., att_type="whole", bn='no', nat=1, name="Real"):
        '''
        Paper https://openreview.net/forum?id=BJgr4kSFDS
        Modified from https://github.com/hyren/query2box/blob/master/codes/model.py
        '''
        super(AttentionSet, self).__init__()

        self.att_reg = att_reg
        self.att_type = att_type
        self.att_tem = att_tem
        self.Attention_module = Attention(mode_dims, att_type=att_type, bn=bn, nat=nat, name = name)

    def forward(self, embeds):
        '''
        Args:
            embeds: shape (B, mode_dims, L), B: batch_size; L: number of embeddings we need to aggregate
        Return:
            combined: shape (B, mode_dims)
        '''
        # temp: shape (B, 1, L) or (B, mode_dims, L)
        temp = (self.Attention_module(embeds) + self.att_reg)/(self.att_tem+1e-4)
        if self.att_type == 'whole':
            # whole: we combine embeddings with a scalar attention coefficient
            # attens: shape (B, 1, L)
            attens = F.softmax(temp, dim=-1)
            # combined: shape (B, mode_dims, L)
            combined = embeds * attens
            # combined: shape (B, mode_dims)
            combined = torch.sum(combined, dim = -1)
            
        elif self.att_type == 'ele':
            # ele: we combine embeds1 and embeds2 with a vector attention coefficient
            # attens: shape (B, mode_dims, L)
            attens = F.softmax(temp, dim=-1)
            # combined: shape (B, mode_dims, L)
            combined = embeds * attens
            # combined: shape (B, mode_dims)
            combined = torch.sum(combined, dim = -1)

        return combined
            


class Attention(nn.Module):
    def __init__(self, mode_dims, att_type="whole", bn = 'no', nat = 1, name="Real"):
        '''
        Paper https://openreview.net/forum?id=BJgr4kSFDS
        Modified from https://github.com/hyren/query2box/blob/master/codes/model.py
        
        Args:
            mode_dims: the input embedding dimention we need to do attention
            att_type: the type of attention
                whole: we combine embeddings with a scalar attention coefficient
                ele: we combine embedding with a vector attention coefficient
            bn: the type of batch noralization type
                no: no batch norm
                before: batch norm before ReLU
                after:  batch norm after ReLU
            nat: scalar = [1,2,3], the number of attention matrix we want to go through before atten_mats2
        '''
        super(Attention, self).__init__()

        self.bn = bn
        self.nat = nat

        self.atten_mats1 = nn.Parameter(torch.FloatTensor(mode_dims, mode_dims))
        nn.init.xavier_uniform_(self.atten_mats1)
        self.register_parameter("atten_mats1_%s"%name, self.atten_mats1)
        if self.nat >= 2:
            self.atten_mats1_1 = nn.Parameter(torch.FloatTensor(mode_dims, mode_dims))
            nn.init.xavier_uniform_(self.atten_mats1_1)
            self.register_parameter("atten_mats1_1_%s"%name, self.atten_mats1_1)
        if self.nat >= 3:
            self.atten_mats1_2 = nn.Parameter(torch.FloatTensor(mode_dims, mode_dims))
            nn.init.xavier_uniform_(self.atten_mats1_2)
            self.register_parameter("atten_mats1_2_%s"%name, self.atten_mats1_2)
        if bn != 'no':
            self.bn1 = nn.BatchNorm1d(mode_dims)
            self.bn1_1 = nn.BatchNorm1d(mode_dims)
            self.bn1_2 = nn.BatchNorm1d(mode_dims)
        if att_type == 'whole':
            self.atten_mats2 = nn.Parameter(torch.FloatTensor(1, mode_dims))
        elif att_type == 'ele':
            self.atten_mats2 = nn.Parameter(torch.FloatTensor(mode_dims, mode_dims))
        nn.init.xavier_uniform_(self.atten_mats2)
        self.register_parameter("atten_mats2_%s"%name, self.atten_mats2)

    def forward(self, center_embed):
        '''
        Args:
            center_embed: shape (B, mode_dims, L), B: batch_size; L: number of embeddings we need to aggregate
        Return:
            temp3:
                if att_type == 'whole':
                    temp3: shape (B, 1, L)
                elif att_type == 'ele':
                    temp3: shape (B, mode_dims, L)
        '''
        temp1 = center_embed
        if self.nat >= 1:
            # temp2: shape (B, mode_dims, L)
            temp2 = torch.einsum('kc,bcl->bkl', self.atten_mats1, temp1)
            if self.bn == 'no': 
                temp2 = F.relu(temp2)
            elif self.bn == 'before':
                temp2 = F.relu(self.bn1(temp2))
            elif self.bn == 'after':
                temp2 = self.bn1(F.relu(temp2))
            # temp2: shape (B, mode_dims, L)
        if self.nat >= 2:
            temp2 = torch.einsum('kc,bcl->bkl', self.atten_mats1_1, temp2)
            if self.bn == 'no':
                temp2 = F.relu(temp2)
            elif self.bn == 'before':
                temp2 = F.relu(self.bn1_1(temp2))
            elif self.bn == 'after':
                temp2 = self.bn1_1(F.relu(temp2))
            # temp2: shape (B, mode_dims, L)
        if self.nat >= 3:
            temp2 = torch.einsum('kc,bcl->bkl', self.atten_mats1_2, temp2)
            if self.bn == 'no':
                temp2 = F.relu(temp2)
            elif self.bn == 'before':
                temp2 = F.relu(self.bn1_2(temp2))
            elif self.bn == 'after':
                temp2 = self.bn1_2(F.relu(temp2))
            # temp2: shape (B, mode_dims, L)

        temp3 = torch.einsum('kc,bcl->bkl', self.atten_mats2, temp2)
        '''
        if att_type == 'whole':
            temp3: shape (B, 1, L)
        elif att_type == 'ele':
            temp3: shape (B, mode_dims, L)
        '''
        return temp3