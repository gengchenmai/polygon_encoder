import fiona 
import geopandas as gpd
import pandas as pd

import math
import os
import json
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import random
import re
import requests
import copy

import shapely
from shapely.ops import transform

from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.linestring import LineString
from shapely.geometry.polygon import LinearRing
from shapely.geometry import box

from fiona.crs import from_epsg


from datetime import datetime

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


from polygonembed.module import *
from polygonembed.SpatialRelationEncoder import *
from polygonembed.resnet import *
from polygonembed.ddsl_utils import *
from polygonembed.ddsl import *
from polygonembed.spec_pool import *





class PolygonEncoder(nn.Module):
    """
    Given a tensor of polygons with shape (batch_size, num_vert+1, coord_dim = 2), encode them into polygon embeddings

    """
    def __init__(self, spa_enc, pgon_seq_enc, spa_embed_dim, pgon_embed_dim, device = "cpu"):
        """
        Args:
            spa_enc: one kind of SpatialRelationEncoder()
            pgon_seq_enc: encoder used for encode a sequence of location embeddings to represent the polygon
            spa_embed_dim: the spatial/point embedding dimention
            pgon_embed_dim: the output polygon embedding dimention
            device:
        """
        super(PolygonEncoder, self).__init__()
        self.spa_enc = spa_enc
        self.pgon_seq_enc = pgon_seq_enc
        # self.num_vert = num_vert
        self.spa_embed_dim = spa_embed_dim
        self.pgon_embed_dim = pgon_embed_dim
        self.device = device

        assert self.pgon_seq_enc.in_channels  == self.spa_embed_dim
        assert self.pgon_seq_enc.out_channels == self.pgon_embed_dim


        
    def forward(self, polygons, V = None, E = None):
        # def forward(self, polygons, do_polygon_random_start = False):
        """
        Args:
            polygons: shape (batch_size, num_vert, coord_dim = 2) 
                note that in num_vert dimention, the last point is not the same as the 1st point
            V: torch.FloatTensor, shape (batch_size, num_vert, 2). vertex coordinates
            E: torch.LongTensor, shape (batch_size, num_vert, 2). vertex connection, edge
        Return:
            pgon_embeds: shape (batch_size, pgon_embed_dim)
        """
        
        if V is not None and E is not None:
            # complex polygon, we just treat V as the polygon vertices sequence
            polygons = V
        
        batch_size, num_vert, coord_dim = polygons.shape

        assert coord_dim == 2

        

        polygons = polygons.cpu().numpy()

        # polygon_spa_embeds: (batch_size, num_vert, spa_embed_dim)
        polygon_spa_embeds = self.spa_enc(polygons)

        # # polygon_spa_embeds_norm: (batch_size, num_vert, 1)
        # polygon_spa_embeds_norm = torch.norm(polygon_spa_embeds, dim = 2, keepdim = True)

        # # polygon_spa_embeds_n: (batch_size, num_vert, spa_embed_dim)
        # polygon_spa_embeds_n = polygon_spa_embeds.div(polygon_spa_embeds_norm)

        # polygon_spa_embeds_: (batch_size, spa_embed_dim, num_vert)
        polygon_spa_embeds_ = polygon_spa_embeds.permute(0, 2, 1)


        # pgon_embeds: shape (batch_size, pgon_embed_dim)
        pgon_embeds = self.pgon_seq_enc(polygon_spa_embeds_)

        # # pgon_embeds_norm: shape (batch_size, 1)
        # pgon_embeds_norm = torch.norm(pgon_embeds, p = 2, dim = 1, keepdim = True)

        # pgon_embeds = pgon_embeds.div(pgon_embeds_norm)





        return pgon_embeds







class VeerCNNPolygonEncoder(nn.Module):
    """
    This is the POlygon CNN method proposed by R.H. van ’t Veer in https://arxiv.org/pdf/1806.03857.pdf
    See Figure 3
    This is a PyTorch reimplementation of https://github.com/SPINLab/geometry-learning/blob/develop/model/building_convnet.py 
    Line 127 - 134

    Given a tensor of polygons with shape (batch_size, num_vert+1, coord_dim = 2), encode them into polygon embeddings

    """
    def __init__(self, pgon_embed_dim, dropout_rate, padding_mode = "zeros", device = "cpu"):
        """
        Args:
            
            pgon_embed_dim: the output polygon embedding dimention
            device:
        """
        super(VeerCNNPolygonEncoder, self).__init__()
        
        self.spa_embed_dim = 5
        self.pgon_embed_dim = pgon_embed_dim
        self.device = device

        self.conv1 = nn.Conv1d(in_channels = self.spa_embed_dim, out_channels = 32, 
                            kernel_size = 5, stride = 1, 
                            padding = 2, padding_mode = padding_mode, bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool1d(kernel_size = 3, stride = 3, padding = 0, ceil_mode = True)

        self.conv2 = nn.Conv1d(in_channels = 32, out_channels = 64, 
                            kernel_size = 5, stride = 1, 
                            padding = 2, padding_mode = padding_mode, bias=True)

        self.linear = nn.Linear(64, self.pgon_embed_dim, bias=True)

        self.dropout = nn.Dropout(p=dropout_rate)

        '''
        Keras default tf.keras.layers.Conv1D() and tf.keras.layers.Dense() -> kernel_initializer="glorot_uniform"
        is equal to nn.init.xavier_uniform_() in pytorch
        see https://keras.io/api/layers/initializers/
        '''
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.linear.weight)
        '''
        Keras default tf.keras.layers.Conv1D() and tf.keras.layers.Dense() -> bias_initializer="zeros"
        is equal to nn.init.zeros_() in pytorch
        see https://keras.io/api/layers/initializers/
        '''
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.linear.bias)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def compute_spa_embed(self, polygons, V = None, E = None):
        """
        add one end point, and add the 3 dim one hot vec as Veer et al did: https://arxiv.org/pdf/1806.03857.pdf
        Args:
            polygons: shape (batch_size, num_vert, coord_dim = 2) 
                note that in num_vert dimention, the last point is not the same as the 1st point
            V: torch.FloatTensor, shape (batch_size, num_vert, 2). vertex coordinates
            E: torch.LongTensor, shape (batch_size, num_vert, 2). vertex connection, edge
        Return:
            polygon_spa_embeds: (batch_size, 5, num_vert + 1 = L)
        """
        if V is not None and E is not None:
            # complex polygon, we just treat V as the polygon vertices sequence
            polygons = V
        
        batch_size, num_vert, coord_dim = polygons.shape

        assert coord_dim == 2


        # 1. add the first point as a loop
        # polygons_: shape (batch_size, num_vert+1, coord_dim = 2)
        polygons_ = torch.cat( [ polygons, polygons[:, 0, :].unsqueeze(1) ], dim=1)

        # 2. make the one hot vector for each point 
        # middle point: [1, 0, 0]
        # end point: [0, 0 , 1]
        # one_hot_vecs: shape (batch_size, num_vert + 1, 3)
        one_hot_vecs = torch.zeros(batch_size, num_vert + 1, 3).to(self.device)
        one_hot_vecs[:, :-1, 0] = 1
        one_hot_vecs[:, -1, 2] = 1

        # polygon_spa_embeds_: shape (batch_size, num_vert + 1, 5)
        polygon_spa_embeds_ = torch.cat([polygons_, one_hot_vecs], dim = -1)


        # polygon_spa_embeds: (batch_size, 5, num_vert + 1 = L)
        polygon_spa_embeds = polygon_spa_embeds_.permute(0, 2, 1)
        return polygon_spa_embeds

    def forward(self, polygons, V = None, E = None):
        # def forward(self, polygons, do_polygon_random_start = False):
        """
        Args:
            polygons: shape (batch_size, num_vert, coord_dim = 2) 
                note that in num_vert dimention, the last point is not the same as the 1st point
            V: torch.FloatTensor, shape (batch_size, num_vert, 2). vertex coordinates
            E: torch.LongTensor, shape (batch_size, num_vert, 2). vertex connection, edge
        Return:
            pgon_embeds: shape (batch_size, pgon_embed_dim)
        """
        # polygon_spa_embeds: (batch_size, 5, num_vert + 1 = L)
        polygon_spa_embeds = self.compute_spa_embed(polygons, V = V, E = E)


        # 2. CNN layer
        # 2.1 Conv1d
        # x: shape (batch_size, 32, L)
        x = self.conv1(polygon_spa_embeds)
        x = self.relu(x)

        # 2.2 Maxpooling
        # x: shape (batch_size, 32, L/3)
        x = self.maxpool1(x)

        # 2.3 Conv1d
        # x: shape (batch_size, 64, L/3)
        x = self.conv2(x)
        x = self.relu(x)

        # 2.4 Global Avergae Pooling
        # x: shape (batch_size, 64)
        x = torch.mean(x, dim = -1, keepdim=False)

        # 2.5 Linear
        # x: shape (batch_size, pgon_embed_dim)
        x = self.linear(x)
        x = self.relu(x)

        # 2.6 dropout
        x = self.dropout(x)

        pgon_embeds = x
        return pgon_embeds





class NUFTPolygonEncoder(nn.Module):
    """
    This mode use Non-uniform discrete Fourier transform (NUFT) to transofrom a polygon into specture domain
    It is modified from DDSL based method modified from https://arxiv.org/pdf/1901.11082.pdf

    Given a tensor of polygons with shape (batch_size, num_vert+1, coord_dim = 2), encode them into polygon embeddings

    """
    def __init__(self, pgon_embed_dim, dropout_rate, ffn, nuft_mode = "ddsl", extent = (-1,1,-1,1), eps = 1e-6,
                 freqXY = [16, 16], 
                 min_freqXY = 1, max_freqXY= 16, mid_freqXY = None, freq_init = "fft",
                 j = 2, embed_norm = "none",
                 smoothing='gaussian', fft_sigma=2.0, elem_batch=100, mode='density', device = "cpu"):
        """
        Args:
            
            pgon_embed_dim: the output polygon embedding dimention
            dropout_rate:
            ffn: MultiLayerFeedForwardNN() 
            extent: the maximum spatial extent of all polygons, (minx, maxx, miny, maxy)
            eps: the noise add to each vertice to make the NUFT more stable
            freqXY: res in DDSL_spec(), [fx, fy]
                fx, fy: number of frequency in X or Y dimention
            max_freqXY: the maximum frequency
            min_freqXY: the minimum frequency
            mid_freqXY: the middle frequency
            freq_init: frequency generated method, 
                "geometric": geometric series
                "fft": fast fourier transformation

            j: simplex dimention
                0: point
                1: line
                2: triangle, polygon can be seem asa  2-simplex mesh, this is the default
            smoothing: str, choice of spectral smoothing function. Defaults 'gaussian'
            fft_sigma: sigma of gaussian at highest frequency
            elem_batch: element-wise batch size.
            mode: normalization mode.
                 'density' for preserving density, 'mass' for preserving mass
            device:
        """
        super(NUFTPolygonEncoder, self).__init__()
        self.pgon_embed_dim = pgon_embed_dim
        self.dropout_rate = dropout_rate
        self.extent = extent
        self.device = device
        self.eps = eps

        self.embed_norm = embed_norm
        
        self.ffn = ffn
        self.pgon_nuft_embed_dim =  ffn.input_dim
        assert self.pgon_embed_dim == ffn.output_dim
        
        self.freqXY = freqXY
        assert len(self.freqXY) ==  2
        self.periodXY = make_periodXY(extent)

        self.min_freqXY = min_freqXY
        self.max_freqXY = max_freqXY
        self.mid_freqXY = mid_freqXY
        # print(min_freqXY, max_freqXY, mid_freqXY)
        self.freq_init = freq_init
        
        self.j = j
        
        self.smoothing = smoothing
        self.fft_sigma = fft_sigma
        self.elem_batch = elem_batch
        self.mode = mode
        
        
        self.ddsl_spec = DDSL_spec(res = self.freqXY, 
                                   t = self.periodXY, 
                                   j = self.j, 
                                   min_freqXY = self.min_freqXY, 
                                   max_freqXY = self.max_freqXY, 
                                   mid_freqXY = self.mid_freqXY,
                                   freq_init = self.freq_init,
                                   elem_batch = elem_batch, 
                                   mode = mode)
        self.ddsl_phys = DDSL_phys(res = self.freqXY, 
                                   t = self.periodXY, 
                                   j = self.j, 
                                   min_freqXY = self.min_freqXY, 
                                   max_freqXY = self.max_freqXY, 
                                   mid_freqXY = self.mid_freqXY,
                                   freq_init = self.freq_init, 
                                   smoothing = smoothing, 
                                   sig = fft_sigma, 
                                   elem_batch = elem_batch, 
                                   mode = mode)
        
        if self.embed_norm == "bn":
            self.batchnorm = nn.BatchNorm1d(num_features = self.pgon_nuft_embed_dim, 
                                    eps=1e-05, 
                                    momentum=0.1, 
                                    affine=True, 
                                    track_running_stats=True)
        
    

    
    def forward(self, polygons, V = None, E = None):
        """
        Args:
            polygons: shape (batch_size, num_vert, coord_dim = 2) 
                note that in num_vert dimention, the last point is not the same as the 1st point
            V: torch.FloatTensor, shape (batch_size, num_vert, 2). vertex coordinates
            E: torch.LongTensor, shape (batch_size, num_vert, 2). vertex connection, edge
        Return:
            pgon_embeds: shape (batch_size, pgon_embed_dim)
        """
        V, E, D = polygon_nuft_input(polygons, self.extent, V, E)

        # F: torch.FloatTensor(), shape (batch_size, fx, fy//2+1, n_channel = 1, 2)
        F = self.ddsl_spec(V, E, D)
        
        
        batch_size = F.shape[0]
        F_type = polygons.dtype if polygons is not None else V.dtype
        # pgon_nuft_embeds: shape (batch_size, fx * (fy//2+1) * n_channel * 2)
        pgon_nuft_embeds = F.reshape(batch_size, -1).to(F_type)

        pgon_nuft_embeds[pgon_nuft_embeds.isnan()] = 0


        if self.embed_norm == "none" or self.embed_norm == "F":
            pgon_nuft_embeds_ = pgon_nuft_embeds
        elif self.embed_norm == "l2":
            # pgon_nuft_embeds_norm: shape (batch_size, 1)
            pgon_nuft_embeds_norm = torch.norm(pgon_nuft_embeds, p = 2, dim = -1, keepdim = True )
            # pgon_nuft_embeds: shape (batch_size, fx * (fy//2+1) * n_channel * 2)
            pgon_nuft_embeds_ = torch.div(pgon_nuft_embeds, pgon_nuft_embeds_norm)
        elif self.embed_norm == "bn":
            pgon_nuft_embeds_ = self.batchnorm(pgon_nuft_embeds)

        
        # pgon_embeds: shape (batch_size, pgon_embed_dim)
        pgon_embeds = self.ffn(pgon_nuft_embeds_)
        return pgon_embeds


class NUFTPCAPolygonEncoder(nn.Module):
    """
    This mode use Non-uniform discrete Fourier transform (NUFT) to transofrom a polygon into specture domain
    It is modified from DDSL based method modified from https://arxiv.org/pdf/1901.11082.pdf

    Given a tensor of polygons with shape (batch_size, num_vert+1, coord_dim = 2), encode them into polygon embeddings

    """
    def __init__(self, pgon_embed_dim, dropout_rate, ffn, nuft_mode = "ddsl", extent = (-1,1,-1,1), eps = 1e-6,
                 freqXY = [16, 16], 
                 min_freqXY = 1, max_freqXY= 16, mid_freqXY = None, freq_init = "fft",
                 j = 2, embed_norm = "none",
                 smoothing='gaussian', fft_sigma=2.0, elem_batch=100, mode='density', device = "cpu",
                 nuft_pca_dim = 108, pca_mat = None):
        """
        Args:
            
            pgon_embed_dim: the output polygon embedding dimention
            dropout_rate:
            ffn: MultiLayerFeedForwardNN() 
            extent: the maximum spatial extent of all polygons, (minx, maxx, miny, maxy)
            eps: the noise add to each vertice to make the NUFT more stable
            freqXY: res in DDSL_spec(), [fx, fy]
                fx, fy: number of frequency in X or Y dimention
            max_freqXY: the maximum frequency
            min_freqXY: the minimum frequency
            freq_init: frequency generated method, 
                "geometric": geometric series
                "fft": fast fourier transformation

            j: simplex dimention
                0: point
                1: line
                2: triangle, polygon can be seem asa  2-simplex mesh, this is the default
            smoothing: str, choice of spectral smoothing function. Defaults 'gaussian'
            fft_sigma: sigma of gaussian at highest frequency
            elem_batch: element-wise batch size.
            mode: normalization mode.
                 'density' for preserving density, 'mass' for preserving mass
            device:
            nuft_pca_dim: number of PCA component we will deduct to
            pca_mat: a precomputed PCA matrix, shape (n_features, max_nuft_pca_dim), 
                    max_nuft_pca_dim == n_features is the maximum possible nuft_pca_dim we can use
                    n_feature = fx * (fy//2+1) * 1 * 2
        """
        super(NUFTPCAPolygonEncoder, self).__init__()
        self.pgon_embed_dim = pgon_embed_dim
        self.dropout_rate = dropout_rate
        self.extent = extent
        self.device = device
        self.eps = eps

        self.embed_norm = embed_norm
        
        self.ffn = ffn
        self.pgon_nuft_embed_dim =  ffn.input_dim
        assert self.pgon_embed_dim == ffn.output_dim
        
        self.freqXY = freqXY
        assert len(self.freqXY) ==  2
        self.periodXY = make_periodXY(extent)

        self.min_freqXY = min_freqXY
        self.max_freqXY = max_freqXY
        self.mid_freqXY = mid_freqXY
        self.freq_init = freq_init
        
        self.j = j
        
        self.smoothing = smoothing
        self.fft_sigma = fft_sigma
        self.elem_batch = elem_batch
        self.mode = mode

        self.nuft_pca_dim = nuft_pca_dim
        self.pca_mat = pca_mat

        self.make_pca_tensor()
        
        
        self.ddsl_spec = DDSL_spec(res = self.freqXY, 
                                   t = self.periodXY, 
                                   j = self.j, 
                                   min_freqXY = self.min_freqXY, 
                                   max_freqXY = self.max_freqXY, 
                                   mid_freqXY = self.mid_freqXY,
                                   freq_init = self.freq_init,
                                   elem_batch = elem_batch, 
                                   mode = mode)
        self.ddsl_phys = DDSL_phys(res = self.freqXY, 
                                   t = self.periodXY, 
                                   j = self.j, 
                                   min_freqXY = self.min_freqXY, 
                                   max_freqXY = self.max_freqXY, 
                                   mid_freqXY = self.mid_freqXY,
                                   freq_init = self.freq_init, 
                                   smoothing = smoothing, 
                                   sig = fft_sigma, 
                                   elem_batch = elem_batch, 
                                   mode = mode)
        
        if self.embed_norm == "bn":
            self.batchnorm = nn.BatchNorm1d(num_features = self.pgon_nuft_embed_dim, 
                                    eps=1e-05, 
                                    momentum=0.1, 
                                    affine=True, 
                                    track_running_stats=True)
        
    def make_pca_tensor(self):
        n_features, n_components = self.pca_mat.shape
        fx, fy = self.freqXY
        self.F_dim = fx * (fy//2+1) * 1 * 2
        assert n_features == self.F_dim
        assert self.nuft_pca_dim <= n_components

        # pca_w: shape (F_dim = fx * (fy//2+1) * 1 * 2, nuft_pca_dim)
        self.pca_w = torch.tensor(data = self.pca_mat[:, :self.nuft_pca_dim], 
                            dtype=torch.float32, 
                            device=self.device, 
                            requires_grad=False, 
                            pin_memory=False)

    
    def forward(self, polygons, V = None, E = None):
        """
        Args:
            polygons: shape (batch_size, num_vert, coord_dim = 2) 
                note that in num_vert dimention, the last point is not the same as the 1st point
            V: torch.FloatTensor, shape (batch_size, num_vert, 2). vertex coordinates
            E: torch.LongTensor, shape (batch_size, num_vert, 2). vertex connection, edge
        Return:
            pgon_embeds: shape (batch_size, pgon_embed_dim)
        """
        V, E, D = polygon_nuft_input(polygons, self.extent, V, E)

        # F: torch.FloatTensor(), shape (batch_size, fx, fy//2+1, n_channel = 1, 2)
        F = self.ddsl_spec(V, E, D)
        
        
        batch_size = F.shape[0]
        # pad nans to 0
        F[torch.isnan(F)] = 0

        F_type = polygons.dtype if polygons is not None else V.dtype
        # F_flat: shape (batch_size, fx * (fy//2+1) * n_channel * 2)
        F_flat = F.reshape(batch_size, -1).to(F_type)

        # pgon_nuft_embeds: shape (batch_size, nuft_pca_dim)
        pgon_nuft_embeds = torch.matmul(F_flat, self.pca_w)



        if self.embed_norm == "none" or self.embed_norm == "F":
            pgon_nuft_embeds_ = pgon_nuft_embeds
        elif self.embed_norm == "l2":
            # pgon_nuft_embeds_norm: shape (batch_size, 1)
            pgon_nuft_embeds_norm = torch.norm(pgon_nuft_embeds, p = 2, dim = -1, keepdim = True )
            # pgon_nuft_embeds: shape (batch_size, nuft_pca_dim)
            pgon_nuft_embeds_ = torch.div(pgon_nuft_embeds, pgon_nuft_embeds_norm)
        elif self.embed_norm == "bn":
            pgon_nuft_embeds_ = self.batchnorm(pgon_nuft_embeds)

        
        # pgon_embeds: shape (batch_size, pgon_embed_dim)
        pgon_embeds = self.ffn(pgon_nuft_embeds_)
        return pgon_embeds


class NUFTSpecPoolPolygonEncoder(nn.Module):
    """
    This mode use Non-uniform discrete Fourier transform (NUFT) to transofrom a polygon into specture domain
    It is modified from DDSL based method modified from https://arxiv.org/pdf/1901.11082.pdf

    We do an extra spetural pooling between the NUFT and FFT based on https://arxiv.org/pdf/1506.03767.pdf

    Given a tensor of polygons with shape (batch_size, num_vert+1, coord_dim = 2), encode them into polygon embeddings

    """
    def __init__(self, pgon_embed_dim, dropout_rate, ffn, nuft_mode = "ddsl", extent = (-1,1,-1,1), eps = 1e-6,
                 freqXY = [16, 16], 
                 min_freqXY = 1, max_freqXY= 16, mid_freqXY = None, freq_init = "fft",
                 j = 2, embed_norm = "none",
                 smoothing='gaussian', fft_sigma=2.0, elem_batch=100, mode='density', 
                 spec_pool_max_freqXY = [16, 16], spec_pool_min_freqXY_ratio = 0.5,
                 device = "cpu"):
        """
        Args:
            
            pgon_embed_dim: the output polygon embedding dimention
            dropout_rate:
            ffn: MultiLayerFeedForwardNN() 
            extent: the maximum spatial extent of all polygons, (minx, maxx, miny, maxy)
            eps: the noise add to each vertice to make the NUFT more stable
            freqXY: res in DDSL_spec(), [fx, fy]
                fx, fy: number of frequency in X or Y dimention
            max_freqXY: the maximum frequency
            min_freqXY: the minimum frequency
            freq_init: frequency generated method, 
                "geometric": geometric series
                "fft": fast fourier transformation

            j: simplex dimention
                0: point
                1: line
                2: triangle, polygon can be seem asa  2-simplex mesh, this is the default
            smoothing: str, choice of spectral smoothing function. Defaults 'gaussian'
            fft_sigma: sigma of gaussian at highest frequency
            elem_batch: element-wise batch size.
            mode: normalization mode.
                 'density' for preserving density, 'mass' for preserving mass
            device:

            spec_pool_max_freqXY:, [wx, wy], the meaningful frequency dimention of fx and fy
        """
        super(NUFTSpecPoolPolygonEncoder, self).__init__()
        self.pgon_embed_dim = pgon_embed_dim
        self.dropout_rate = dropout_rate
        self.extent = extent
        self.device = device
        self.eps = eps

        self.embed_norm = embed_norm
        
        self.ffn = ffn
        self.pgon_nuft_embed_dim =  ffn.input_dim
        assert self.pgon_embed_dim == ffn.output_dim
        
        self.freqXY = freqXY
        assert len(self.freqXY) ==  2
        self.periodXY = make_periodXY(extent)

        self.min_freqXY = min_freqXY
        self.max_freqXY = max_freqXY
        self.mid_freqXY = mid_freqXY
        self.freq_init = freq_init
        
        self.j = j
        
        self.smoothing = smoothing
        self.fft_sigma = fft_sigma
        self.elem_batch = elem_batch
        self.mode = mode
        
        
        self.ddsl_spec = DDSL_spec(res = self.freqXY, 
                                   t = self.periodXY, 
                                   j = self.j, 
                                   min_freqXY = self.min_freqXY, 
                                   max_freqXY = self.max_freqXY, 
                                   mid_freqXY = self.mid_freqXY,
                                   freq_init = self.freq_init,
                                   elem_batch = elem_batch, 
                                   mode = mode)
        self.ddsl_phys = DDSL_phys(res = self.freqXY, 
                                   t = self.periodXY, 
                                   j = self.j, 
                                   min_freqXY = self.min_freqXY, 
                                   max_freqXY = self.max_freqXY, 
                                   mid_freqXY = self.mid_freqXY,
                                   freq_init = self.freq_init, 
                                   smoothing = smoothing, 
                                   sig = fft_sigma, 
                                   elem_batch = elem_batch, 
                                   mode = mode)

        self.spec_pool_max_freqXY = spec_pool_max_freqXY
        self.spec_pool_min_freqXY_ratio = spec_pool_min_freqXY_ratio
        # self.spec_pool = SpecturalPooling(
        #                             min_freqXY_ratio = spec_pool_min_freqXY_ratio, 
        #                             max_pool_freqXY = spec_pool_max_freqXY, 
        #                             freqXY = freqXY, 
        #                             device = device)
        
        if self.embed_norm == "bn":
            self.batchnorm = nn.BatchNorm1d(num_features = self.pgon_nuft_embed_dim, 
                                    eps=1e-05, 
                                    momentum=0.1, 
                                    affine=True, 
                                    track_running_stats=True)
        
    

    
    def forward(self, polygons,  V = None, E = None):
        """
        Args:
            polygons: shape (batch_size, num_vert, coord_dim = 2) 
                note that in num_vert dimention, the last point is not the same as the 1st point
        Return:
            pgon_embeds: shape (batch_size, pgon_embed_dim)
        """
        V, E, D = polygon_nuft_input(polygons, self.extent, V = V, E = E)

        # F: torch.FloatTensor(), shape (batch_size, fx, fy//2+1, n_channel = 1, 2)
        F = self.ddsl_spec(V, E, D)
        
        
        batch_size = F.shape[0]
        # pad nans to 0
        F[torch.isnan(F)] = 0
        F_type = polygons.dtype if polygons is not None else V.dtype

        wx, wy = self.spec_pool_max_freqXY
        # F1: shape (batch_size, wx, wy, n_channel = 1, 2)
        F1 = F[:, :wx, :wy, :, :]
        # F2: shape (batch_size, wx, wy, n_channel = 1, 2)
        F2 = F[:, -wx:, :wy, :, :]
        # F_pool: shape (batch_size, 2*wx, wy, n_channel = 1, 2)
        F_pool = torch.cat([F1, F2], dim = 1)

        # pgon_nuft_embeds: shape (batch_size, 4 * wx * wy)
        pgon_nuft_embeds = F_pool.reshape(batch_size, -1).to(F_type)



        if self.embed_norm == "none":
            pgon_nuft_embeds_ = pgon_nuft_embeds
        elif self.embed_norm == "l2":
            # pgon_nuft_embeds_norm: shape (batch_size, 1)
            pgon_nuft_embeds_norm = torch.norm(pgon_nuft_embeds, p = 2, dim = -1, keepdim = True )
            # pgon_nuft_embeds: shape (batch_size, max_fx * ceil(max_fy/2) * n_channel * 2)
            pgon_nuft_embeds_ = torch.div(pgon_nuft_embeds, pgon_nuft_embeds_norm)
        elif self.embed_norm == "bn":
            pgon_nuft_embeds_ = self.batchnorm(pgon_nuft_embeds)

        
        # pgon_embeds: shape (batch_size, pgon_embed_dim)
        pgon_embeds = self.ffn(pgon_nuft_embeds_)
        return pgon_embeds

# class NUFTSpecPoolPolygonEncoder(nn.Module):
#     """
#     This mode use Non-uniform discrete Fourier transform (NUFT) to transofrom a polygon into specture domain
#     It is modified from DDSL based method modified from https://arxiv.org/pdf/1901.11082.pdf

#     We do an extra spetural pooling between the NUFT and FFT based on https://arxiv.org/pdf/1506.03767.pdf

#     Given a tensor of polygons with shape (batch_size, num_vert+1, coord_dim = 2), encode them into polygon embeddings

#     """
#                  smoothing='gaussian', fft_sigma=2.0, elem_batch=100, mode='density', device = "cpu"
#     def __init__(self, pgon_embed_dim, dropout_rate, ffn, nuft_mode = "ddsl", extent = (-1,1,-1,1), eps = 1e-6,
#                  freqXY = [16, 16], j = 2, embed_norm = "none",
#                  smoothing='gaussian', fft_sigma=2.0, elem_batch=100, mode='density', 
#                  spec_pool_max_freqXY = [16, 16], spec_pool_min_freqXY_ratio = 0.5,
#                  device = "cpu"):
#         """
#         Args:
            
#             pgon_embed_dim: the output polygon embedding dimention
#             dropout_rate:
#             ffn: MultiLayerFeedForwardNN() 
#             extent: the maximum spatial extent of all polygons, (minx, maxx, miny, maxy)
#             eps: the noise add to each vertice to make the NUFT more stable
#             freqXY: res in DDSL_spec(), [fx, fy]
#                 fx, fy: number of frequency in X or Y dimention
            
#             j: simplex dimention
#                 0: point
#                 1: line
#                 2: triangle, polygon can be seem asa  2-simplex mesh, this is the default
#             smoothing: str, choice of spectral smoothing function. Defaults 'gaussian'
#             fft_sigma: sigma of gaussian at highest frequency
#             elem_batch: element-wise batch size.
#             mode: normalization mode.
#                  'density' for preserving density, 'mass' for preserving mass
#             device:
#         """
#         super(NUFTSpecPoolPolygonEncoder, self).__init__()
#         self.pgon_embed_dim = pgon_embed_dim
#         self.dropout_rate = dropout_rate
#         self.extent = extent
#         self.device = device
#         self.eps = eps

#         self.embed_norm = embed_norm
        
#         self.ffn = ffn
#         self.pgon_nuft_embed_dim =  ffn.input_dim
#         assert self.pgon_embed_dim == ffn.output_dim
        
#         self.freqXY = freqXY
#         assert len(self.freqXY) ==  2
#         self.periodXY = make_periodXY(extent)
        
#         self.j = j
        
#         self.smoothing = smoothing
#         self.fft_sigma = fft_sigma
#         self.elem_batch = elem_batch
#         self.mode = mode
        
        
#         self.ddsl_spec = DDSL_spec(res = self.freqXY, 
#                                    t = self.periodXY, 
#                                    j = self.j, 
#                                    elem_batch = elem_batch, 
#                                    mode = mode)
#         self.ddsl_phys = DDSL_phys(res = self.freqXY, 
#                                    t = self.periodXY, 
#                                    j = self.j, 
#                                    smoothing = smoothing, 
#                                    sig = fft_sigma, 
#                                    elem_batch = elem_batch, 
#                                    mode = mode)

#         self.spec_pool_max_freqXY = spec_pool_max_freqXY
#         self.spec_pool_min_freqXY_ratio = spec_pool_min_freqXY_ratio
#         self.spec_pool = SpecturalPooling(
#                                     min_freqXY_ratio = spec_pool_min_freqXY_ratio, 
#                                     max_pool_freqXY = spec_pool_max_freqXY, 
#                                     freqXY = freqXY, 
#                                     device = device)
        
#         if self.embed_norm == "bn":
#             self.batchnorm = nn.BatchNorm1d(num_features = self.pgon_nuft_embed_dim, 
#                                     eps=1e-05, 
#                                     momentum=0.1, 
#                                     affine=True, 
#                                     track_running_stats=True)
        
    

    
#     def forward(self, polygons,  V = None, E = None):
#         """
#         Args:
#             polygons: shape (batch_size, num_vert, coord_dim = 2) 
#                 note that in num_vert dimention, the last point is not the same as the 1st point
#         Return:
#             pgon_embeds: shape (batch_size, pgon_embed_dim)
#         """
#         V, E, D = polygon_nuft_input(polygons, self.extent, V = V, E = E)

#         # F: torch.FloatTensor(), 
#         F = self.ddsl_spec(V, E, D)
        
        
#         batch_size = F.shape[0]

#         # x: shape (batch_size, n_channel = 1, fx, fy//2+1, 2)
#         x = F.permute(0, 3, 1, 2, 4)

#         # spec_pool_mask_res: shape (batch_size, n_channel = 1, max_fx, ceil(max_fy/2),  2) 
#         spec_pool_mask_res = self.spec_pool(x)

#         F_type = polygons.dtype if polygons is not None else V.dtype
#         # pgon_nuft_embeds: shape (batch_size, max_fx * ceil(max_fy/2) * n_channel * 2)
#         pgon_nuft_embeds = spec_pool_mask_res.reshape(batch_size, -1).to(F_type)

#         # pgon_nuft_embeds[pgon_nuft_embeds.isnan()] = 0


#         if self.embed_norm == "none":
#             pgon_nuft_embeds_ = pgon_nuft_embeds
#         elif self.embed_norm == "l2":
#             # pgon_nuft_embeds_norm: shape (batch_size, 1)
#             pgon_nuft_embeds_norm = torch.norm(pgon_nuft_embeds, p = 2, dim = -1, keepdim = True )
#             # pgon_nuft_embeds: shape (batch_size, max_fx * ceil(max_fy/2) * n_channel * 2)
#             pgon_nuft_embeds_ = torch.div(pgon_nuft_embeds, pgon_nuft_embeds_norm)
#         elif self.embed_norm == "bn":
#             pgon_nuft_embeds_ = self.batchnorm(pgon_nuft_embeds)

        
#         # pgon_embeds: shape (batch_size, pgon_embed_dim)
#         pgon_embeds = self.ffn(pgon_nuft_embeds_)
#         return pgon_embeds



class NUFTIFFTPolygonEncoder(nn.Module):
    """
    This mode use Non-uniform discrete Fourier transform (NUFT) to transofrom a polygon into specture domain
    It is modified from DDSL based method modified from https://arxiv.org/pdf/1901.11082.pdf

    Given a tensor of polygons with shape (batch_size, num_vert+1, coord_dim = 2), encode them into polygon embeddings

    """
    def __init__(self, extent = (-1,1,-1,1), eps = 1e-6,
                freqXY = [16, 16], 
                min_freqXY = 1, max_freqXY= 16, mid_freqXY = None, freq_init = "fft",
                j = 2,
                smoothing = 'gaussian', fft_sigma=2.0, elem_batch=100, mode='density', device = "cpu"):
        """
        Args:
            extent: the maximum spatial extent of all polygons, (minx, maxx, miny, maxy)
            eps: the noise add to each vertice to make the NUFT more stable
            freqXY: res in DDSL_spec(), [fx, fy]
                fx, fy: number of frequency in X or Y dimention
            max_freqXY: the maximum frequency
            min_freqXY: the minimum frequency
            freq_init: frequency generated method, 
                "geometric": geometric series
                "fft": fast fourier transformation

            j: simplex dimention
                0: point
                1: line
                2: triangle, polygon can be seem asa  2-simplex mesh, this is the default
            smoothing: str, choice of spectral smoothing function. Defaults 'gaussian'
            fft_sigma: sigma of gaussian at highest frequency
            elem_batch: element-wise batch size.
            mode: normalization mode.
                 'density' for preserving density, 'mass' for preserving mass
            device:
        """
        super(NUFTIFFTPolygonEncoder, self).__init__()
        
        self.extent = extent
        self.device = device
        self.eps = eps
        
        
        self.freqXY = freqXY
        assert len(self.freqXY) ==  2
        self.periodXY = make_periodXY(extent)


        self.min_freqXY = min_freqXY
        self.max_freqXY = max_freqXY
        self.mid_freqXY = mid_freqXY
        self.freq_init = freq_init
        assert freq_init == "fft"
        
        self.j = j
        
        self.smoothing = smoothing
        self.fft_sigma = fft_sigma
        self.elem_batch = elem_batch
        self.mode = mode
        
        
        self.ddsl_spec = DDSL_spec(res = self.freqXY, 
                                   t = self.periodXY, 
                                   j = self.j, 
                                   min_freqXY = self.min_freqXY, 
                                   max_freqXY = self.max_freqXY, 
                                   mid_freqXY = self.mid_freqXY,
                                   freq_init = self.freq_init,
                                   elem_batch = elem_batch, 
                                   mode = mode)
        self.ddsl_phys = DDSL_phys(res = self.freqXY, 
                                   t = self.periodXY, 
                                   j = self.j, 
                                   min_freqXY = self.min_freqXY, 
                                   max_freqXY = self.max_freqXY, 
                                   mid_freqXY = self.mid_freqXY,
                                   freq_init = self.freq_init, 
                                   smoothing = smoothing, 
                                   sig = fft_sigma, 
                                   elem_batch = elem_batch, 
                                   mode = mode)
        
    
    def forward(self, polygons, V = None, E = None):
        """
        Args:
            polygons: shape (batch_size, num_vert, coord_dim = 2) 
                note that in num_vert dimention, the last point is not the same as the 1st point
        Return:
            pgon_embeds: shape (batch_size, fx, fy, n_channel = 1)
        """
        V, E, D = polygon_nuft_input(polygons, self.extent, V, E)

        # f: shape (batch_size, fx, fy, n_channel = 1)
        f = self.ddsl_phys(V, E, D)

        f_type = polygons.dtype if polygons is not None else V.dtype
        f = f.to(f_type)
        
        return f



class NUFTIFFTMLPPolygonEncoder(nn.Module):
    """
    NUFT + IFFT + MLP

    Given a tensor of polygons with shape (batch_size, num_vert+1, coord_dim = 2), encode them into polygon embeddings

    """
    def __init__(self, pgon_embed_dim, ffn, extent = (-1,1,-1,1), eps = 1e-6,
                 freqXY = [16, 16], 
                 min_freqXY = 1, max_freqXY= 16, mid_freqXY = None, freq_init = "fft",
                 j = 2, embed_norm = "none",
                 smoothing = 'gaussian', fft_sigma=2.0, elem_batch=100, mode='density', 
                 device = "cpu"):
        """
        Args:
            extent: the maximum spatial extent of all polygons, (minx, maxx, miny, maxy)
            eps: the noise add to each vertice to make the NUFT more stable
            freqXY: res in DDSL_spec(), [fx, fy]
                fx, fy: number of frequency in X or Y dimention
            max_freqXY: the maximum frequency
            min_freqXY: the minimum frequency
            freq_init: frequency generated method, 
                "geometric": geometric series
                "fft": fast fourier transformation

            j: simplex dimention
                0: point
                1: line
                2: triangle, polygon can be seem asa  2-simplex mesh, this is the default
            smoothing: str, choice of spectral smoothing function. Defaults 'gaussian'
            fft_sigma: sigma of gaussian at highest frequency
            elem_batch: element-wise batch size.
            mode: normalization mode.
                 'density' for preserving density, 'mass' for preserving mass
            device:
        """
        super(NUFTIFFTMLPPolygonEncoder, self).__init__()
        
        self.extent = extent
        self.device = device
        self.eps = eps
        self.pgon_embed_dim  =pgon_embed_dim
        
        self.freqXY = freqXY
        assert len(self.freqXY) ==  2
        self.periodXY = make_periodXY(extent)

        self.min_freqXY = min_freqXY
        self.max_freqXY = max_freqXY
        self.mid_freqXY = mid_freqXY
        self.freq_init = freq_init
        assert freq_init == "fft"
        
        self.j = j
        
        self.smoothing = smoothing
        self.fft_sigma = fft_sigma
        self.elem_batch = elem_batch
        self.mode = mode

        self.ffn = ffn
        self.pgon_ifft_embed_dim =  ffn.input_dim
        assert self.pgon_embed_dim == ffn.output_dim
        self.embed_norm = embed_norm
        
        
        self.ddsl_spec = DDSL_spec(res = self.freqXY, 
                                   t = self.periodXY, 
                                   j = self.j, 
                                   min_freqXY = self.min_freqXY, 
                                   max_freqXY = self.max_freqXY, 
                                   mid_freqXY = self.mid_freqXY,
                                   freq_init = self.freq_init,
                                   elem_batch = elem_batch, 
                                   mode = mode)
        self.ddsl_phys = DDSL_phys(res = self.freqXY, 
                                   t = self.periodXY, 
                                   j = self.j, 
                                   min_freqXY = self.min_freqXY, 
                                   max_freqXY = self.max_freqXY, 
                                   mid_freqXY = self.mid_freqXY,
                                   freq_init = self.freq_init, 
                                   smoothing = smoothing, 
                                   sig = fft_sigma, 
                                   elem_batch = elem_batch, 
                                   mode = mode)

        if self.embed_norm == "bn":
            self.batchnorm = nn.BatchNorm1d(num_features = self.pgon_ifft_embed_dim, 
                                    eps=1e-05, 
                                    momentum=0.1, 
                                    affine=True, 
                                    track_running_stats=True)
        
    
    def forward(self, polygons, V = None, E = None):
        """
        Args:
            polygons: shape (batch_size, num_vert, coord_dim = 2) 
                note that in num_vert dimention, the last point is not the same as the 1st point
        Return:
            pgon_embeds: shape (batch_size, fx, fy, n_channel = 1)
        """
        V, E, D = polygon_nuft_input(polygons, self.extent, V, E)

        # f: shape (batch_size, fx, fy, n_channel = 1)
        f = self.ddsl_phys(V, E, D)

        f_type = polygons.dtype if polygons is not None else V.dtype
        f = f.to(f_type)

        batch_size = f.shape[0]

        # f_feats; shape (batch_size, fx * fy)
        f_feats = f.reshape(batch_size, -1)


        pgon_ifft_embeds = f_feats


        if self.embed_norm == "none" or self.embed_norm == "F":
            pgon_ifft_embeds_ = pgon_ifft_embeds
        elif self.embed_norm == "l2":
            # pgon_ifft_embeds_norm: shape (batch_size, 1)
            pgon_ifft_embeds_norm = torch.norm(pgon_ifft_embeds, p = 2, dim = -1, keepdim = True )
            # pgon_ifft_embeds_: shape (batch_size, nuft_pca_dim)
            pgon_ifft_embeds_ = torch.div(pgon_ifft_embeds, pgon_ifft_embeds_norm)
        elif self.embed_norm == "bn":
            pgon_ifft_embeds_ = self.batchnorm(pgon_ifft_embeds)


        # pgon_embeds: shape (batch_size, pgon_embed_dim)
        pgon_embeds = self.ffn(pgon_ifft_embeds_)
        return pgon_embeds


class NUFTIFFTPCAMLPPolygonEncoder(nn.Module):
    """
    NUFT + IFFT + MLP

    Given a tensor of polygons with shape (batch_size, num_vert+1, coord_dim = 2), encode them into polygon embeddings

    """
    def __init__(self, pgon_embed_dim, ffn, extent = (-1,1,-1,1), eps = 1e-6,
                 freqXY = [16, 16], 
                 min_freqXY = 1, max_freqXY= 16, mid_freqXY = None, freq_init = "fft",
                 j = 2, embed_norm = "none",
                 smoothing = 'gaussian', fft_sigma=2.0, elem_batch=100, mode='density', 
                 pca_dim = 108, pca_mat = None,
                 device = "cpu"):
        """
        Args:
            extent: the maximum spatial extent of all polygons, (minx, maxx, miny, maxy)
            eps: the noise add to each vertice to make the NUFT more stable
            freqXY: res in DDSL_spec(), [fx, fy]
                fx, fy: number of frequency in X or Y dimention
            max_freqXY: the maximum frequency
            min_freqXY: the minimum frequency
            freq_init: frequency generated method, 
                "geometric": geometric series
                "fft": fast fourier transformation

            j: simplex dimention
                0: point
                1: line
                2: triangle, polygon can be seem asa  2-simplex mesh, this is the default
            smoothing: str, choice of spectral smoothing function. Defaults 'gaussian'
            fft_sigma: sigma of gaussian at highest frequency
            elem_batch: element-wise batch size.
            mode: normalization mode.
                 'density' for preserving density, 'mass' for preserving mass
            device:
            ffn: the FFN/MLP after PCA feature extractor
            pca_dim: number of PCA component we will deduct to
            pca_mat: a precomputed PCA matrix, shape (n_features, max_pca_dim), 
                    max_pca_dim == n_features is the maximum possible pca_dim we can use
                    n_feature = fx * fy
        """
        super(NUFTIFFTPCAMLPPolygonEncoder, self).__init__()
        self.pgon_embed_dim = pgon_embed_dim
        self.extent = extent
        self.device = device
        self.eps = eps
        
        
        self.freqXY = freqXY
        assert len(self.freqXY) ==  2
        self.periodXY = make_periodXY(extent)

        self.min_freqXY = min_freqXY
        self.max_freqXY = max_freqXY
        self.mid_freqXY = mid_freqXY
        self.freq_init = freq_init
        assert freq_init == "fft"
        
        self.j = j
        
        self.smoothing = smoothing
        self.fft_sigma = fft_sigma
        self.elem_batch = elem_batch
        self.mode = mode

        self.ffn = ffn
        self.pca_dim = pca_dim
        self.pca_mat = pca_mat

        assert self.pca_dim == ffn.input_dim
        assert self.pgon_embed_dim == ffn.output_dim
        self.embed_norm = embed_norm

        

        self.make_pca_tensor()
        
        
        self.ddsl_spec = DDSL_spec(res = self.freqXY, 
                                   t = self.periodXY, 
                                   j = self.j, 
                                   min_freqXY = self.min_freqXY, 
                                   max_freqXY = self.max_freqXY, 
                                   mid_freqXY = self.mid_freqXY,
                                   freq_init = self.freq_init,
                                   elem_batch = elem_batch, 
                                   mode = mode)
        self.ddsl_phys = DDSL_phys(res = self.freqXY, 
                                   t = self.periodXY, 
                                   j = self.j, 
                                   min_freqXY = self.min_freqXY, 
                                   max_freqXY = self.max_freqXY, 
                                   mid_freqXY = self.mid_freqXY,
                                   freq_init = self.freq_init, 
                                   smoothing = smoothing, 
                                   sig = fft_sigma, 
                                   elem_batch = elem_batch, 
                                   mode = mode)

        if self.embed_norm == "bn":
            self.batchnorm = nn.BatchNorm1d(num_features = self.pgon_ifft_embed_dim, 
                                    eps=1e-05, 
                                    momentum=0.1, 
                                    affine=True, 
                                    track_running_stats=True)

    def make_pca_tensor(self):
        n_features, n_components = self.pca_mat.shape
        fx, fy = self.freqXY
        self.f_dim = fx * fy
        assert n_features == self.f_dim
        assert self.pca_dim <= n_components

        # pca_w: shape (f_dim = fx * fy, pca_dim)
        self.pca_w = torch.tensor(data = self.pca_mat[:, :self.pca_dim], 
                            dtype=torch.float32, 
                            device=self.device, 
                            requires_grad=False, 
                            pin_memory=False)
        
    
    def forward(self, polygons, V = None, E = None):
        """
        Args:
            polygons: shape (batch_size, num_vert, coord_dim = 2) 
                note that in num_vert dimention, the last point is not the same as the 1st point
        Return:
            pgon_embeds: shape (batch_size, fx, fy, n_channel = 1)
        """
        V, E, D = polygon_nuft_input(polygons, self.extent, V, E)

        # f: shape (batch_size, fx, fy, n_channel = 1)
        f = self.ddsl_phys(V, E, D)

        f_type = polygons.dtype if polygons is not None else V.dtype
        f = f.to(f_type)

        batch_size = f.shape[0]

        # f_feats; shape (batch_size, fx * fy)
        f_feats = f.reshape(batch_size, -1)

        # pgon_pca_embeds: shape (batch_size, pca_dim)
        pgon_ifft_embeds = torch.matmul(f_feats, self.pca_w)


        if self.embed_norm == "none" or self.embed_norm == "F":
            pgon_ifft_embeds_ = pgon_ifft_embeds
        elif self.embed_norm == "l2":
            # pgon_ifft_embeds_norm: shape (batch_size, 1)
            pgon_ifft_embeds_norm = torch.norm(pgon_ifft_embeds, p = 2, dim = -1, keepdim = True )
            # pgon_ifft_embeds_: shape (batch_size, nuft_pca_dim)
            pgon_ifft_embeds_ = torch.div(pgon_ifft_embeds, pgon_ifft_embeds_norm)
        elif self.embed_norm == "bn":
            pgon_ifft_embeds_ = self.batchnorm(pgon_ifft_embeds)

        # pgon_embeds: shape (batch_size, pgon_embed_dim)
        pgon_embeds = self.ffn(pgon_ifft_embeds_)
        return pgon_embeds


class ConcatPolygonEncoder(nn.Module):
    """
    This mode use Non-uniform discrete Fourier transform (NUFT) to transofrom a polygon into specture domain
    It is modified from DDSL based method modified from https://arxiv.org/pdf/1901.11082.pdf

    Given a tensor of polygons with shape (batch_size, num_vert+1, coord_dim = 2), encode them into polygon embeddings

    """
    def __init__(self, pgon_embed_dim, pgon_enc_1, pgon_enc_2, device = "cpu"):
        """
        two polygon encoder, concate their result
        Args:
            pgon_enc_1:
            pgon_enc_2:
            
            device:
        """
        super(ConcatPolygonEncoder, self).__init__()
        self.pgon_embed_dim = pgon_embed_dim
        assert pgon_embed_dim == pgon_enc_1.pgon_embed_dim + pgon_enc_2.pgon_embed_dim
        self.pgon_enc_1 = pgon_enc_1
        self.pgon_enc_2 = pgon_enc_2
        

    
    def forward(self, polygons):
        """
        Args:
            polygons: shape (batch_size, num_vert, coord_dim = 2) 
                note that in num_vert dimention, the last point is not the same as the 1st point
        Return:
            pgon_embeds: shape (batch_size, pgon_embed_dim)
        """
        # pgon_embeds_1: shape (batch_size, pgon_embed_dim_1)
        pgon_embeds_1 = self.pgon_enc_1(polygons)

        # pgon_embeds_2: shape (batch_size, pgon_embed_dim_2)
        pgon_embeds_2 = self.pgon_enc_2(polygons)

        # pgon_embeds: shape (batch_size, pgon_embed_dim)
        pgon_embeds = torch.cat([pgon_embeds_1, pgon_embeds_2], dim = -1)

        return pgon_embeds