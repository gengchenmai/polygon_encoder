import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import math
import numpy as np
from math import pi

from polygonembed.ddsl_utils import *


class SpecturalPooling(nn.Module):
    """
    Do spectural pooling similar to https://arxiv.org/pdf/1506.03767.pdf
    """
    def __init__(self, min_freqXY_ratio = 0.5, max_pool_freqXY = [10, 10], freqXY = [16, 16], device = "cpu"):
        """
        Args:
            min_freqXY_ratio: c_m, the ratio to control the lowest fx, fy to keep, 
                c_m = alpha + m/M * (beta - alpha)
            maxPoolFreqXY: H_m, the maximum frequency we want to keep, 
            freqXY: res in DDSL_spec(), [fx, fy]
                fx, fy: number of frequency in X or Y dimention
            device:
        """
        super(SpecturalPooling, self).__init__()
        self.min_freqXY_ratio = min_freqXY_ratio
        self.max_pool_freqXY = max_pool_freqXY
        self.freqXY = freqXY
        self.device = device

        # the maximum pool fx, fy should smaller/equal to freqXY
        assert len(freqXY) == len(max_pool_freqXY)
        for i in range(len(freqXY)):
            assert freqXY[i] >= max_pool_freqXY[i]

        


    def get_pool_freqXY(self, freqXY_ratio, max_pool_freqXY):
        min_pool_freqXY = []
        for f in max_pool_freqXY:
            min_pool_freqXY.append(math.floor(f * freqXY_ratio))
        return min_pool_freqXY

    def make_select_mask(self, fx, fy, maxfx, maxfy, y_dim):
        '''
        mask the non select freq elements into 0
        Args:
            fx, fy: the select dimention in x, y axis
            maxfx, maxfy: the original freq dimention
            y_dim: the y dimention
        Return:
            mask: torch.FloatTensor(), shape (maxfx, y_dim, 1)

        '''
        assert maxfy > y_dim
        fxtop, fxlow, fytop = self.get_freqXY_select_idx(fx, fy, maxfx, maxfy)
        
        mask = np.zeros((maxfx, y_dim))
        mask[0:fxtop, 0:fytop] = 1
        mask[-fxlow:, 0:fytop] = 1
        
        mask = torch.FloatTensor(mask).unsqueeze(-1)
        return mask

    def get_freqXY_select_idx(self, fx, fy, maxfx, maxfy):
        '''
        mask the non select freq elements into 0
        Args:
            fx, fy: the select dimention in x, y axis
            maxfx, maxfy: the original freq dimention
        Return:
            fxtop, fxlow: 0..fxtop and -fylow ... -1 selected
            fytop: 0...fytop selected

        '''
        fxtop = math.ceil(fx/2)
        fxlow = fx - fxtop

        fytop = math.ceil(fy/2)
        return fxtop, fxlow, fytop

    def crop_freqmap(self, x, cur_pool_freqXY, freqXY):
        '''
        crop the frequency map to (fx, fy)
        Args:
            x: torch.FloatTensor(), the input features (e.g., polygons) in the specture domain
                shape (batch_size, n_channel = 1, maxfx, maxfy//2+1,  2) 
            cur_pool_freqXY: the pool freq dimention
            freqXY: the original dimention
        Return: 
            spec_pool_res: torch.FloatTensor(),
                shape (batch_size, n_channel = 1, fx, ceil(fy/2),  2) 
        '''
        maxfx, maxfy = freqXY
        fx, fy = cur_pool_freqXY
        fxtop, fxlow, fytop = self.get_freqXY_select_idx(fx, fy, maxfx, maxfy)
        
        upblock = x[:, :, 0:fxtop, 0:fytop, :]
        lowblock = x[:, :, -fxlow:, 0:fytop, :]
        
        spec_pool_res = torch.cat([upblock, lowblock], dim = 2)
        return spec_pool_res

    def forward(self, x):
        '''
        Args:
            x: torch.FloatTensor(), the input features (e.g., polygons) in the specture domain
                shape (batch_size, n_channel = 1, fx, fy//2+1,  2) 
        '''
        freqXY_ratio = np.random.uniform(self.min_freqXY_ratio, 1)
        # compute the current no-maked X, Y dimention
        cur_pool_freqXY = self.get_min_pool_freqXY(freqXY_ratio, max_pool_freqXY)

        batch_size, n_channel, fx, fy2,  n_dim = x.shape
        assert fx == self.freqXY[0]
        assert fy2 == self.freqXY[1]//2 + 1

        # crop the input x into max_pool_freqXY
        # spec_pool_res: shape (batch_size, n_channel = 1, max_pool_freqXY[0], ceil(max_pool_freqXY[1]/2),  2) 
        spec_pool_res = self.crop_freqmap(x, max_pool_freqXY, freqXY)

        # mask the freq element outside of cur_pool_freqXY
        # mask:  shape (max_pool_freqXY[0], ceil(max_pool_freqXY[1]/2), 1)
        mask = make_select_mask(fx = cur_pool_freqXY[0], 
                                fy = cur_pool_freqXY[1], 
                                maxfx = max_pool_freqXY[0], 
                                maxfy = max_pool_freqXY[1],
                                y_dim = spec_pool_res.shape[-2])

        # spec_pool_mask_res: shape (batch_size, n_channel = 1, max_pool_freqXY[0], ceil(max_pool_freqXY[1]/2),  2) 
        spec_pool_mask_res = spec_pool_res * mask.to(x.device)
        return spec_pool_mask_res



