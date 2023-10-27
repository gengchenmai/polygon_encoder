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
from polygonembed.utils import *


class ExplicitMLPPolygonDecoder(nn.Module):
    """
    
    """
    def __init__(self, spa_enc, pgon_embed_dim, num_vert, pgon_dec_grid_init = "uniform", pgon_dec_grid_enc_type = "none",
                coord_dim = 2, extent = (-1, 1, -1, 1), device = "cpu"):
        """
        Args:
            spa_enc: a spatial encoder
            pgon_embed_dim: the output polygon embedding dimention
            num_vert: number of uniuqe vertices each generated polygon with have
                note that this does not include the extra point (same as the 1st one) to close the ring
            pgon_dec_grid_init: We generate a list of grid points for polygon decoder, the type of grid points are:
                uniform: points uniformly sampled from (-1, 1, -1, 1) 
                circle: points sampled equal-distance on a circle whose radius is randomly sampled
                kdgrid: k-d regular grid, (num_vert - k^2) is uniformly sampled
            pgon_dec_grid_enc_type: the type to encode the grid point
                none: no encoding, use the original grid point
                spa_enc: use space encoder to encode grid point before 
            coord_dim: 2
            device:
        """
        super(ExplicitMLPPolygonDecoder, self).__init__()
        
        self.spa_enc = spa_enc
        self.spa_embed_dim = self.spa_enc.spa_embed_dim
        self.pgon_embed_dim = pgon_embed_dim
        self.num_vert = num_vert
        self.pgon_dec_grid_init = pgon_dec_grid_init
        self.coord_dim = coord_dim
        self.extent = extent

        self.device = device
        self.grid_dim = 2

        self.pgon_dec_grid_enc_type = pgon_dec_grid_enc_type

        # Partly borrowed from atlasnetV2
        # define point generator 
        if self.pgon_dec_grid_enc_type == "none":
            self.nlatent = self.pgon_embed_dim + self.grid_dim
        elif self.pgon_dec_grid_enc_type == "spa_enc":
            self.nlatent = self.pgon_embed_dim + self.spa_embed_dim
        else:
            raise Exception("Unknown pgon_dec_grid_enc_type")

        # by default the bias = True, so this is like a individual MLP
        self.conv1 = torch.nn.Conv1d(in_channels = self.nlatent,    out_channels = self.nlatent,    kernel_size = 1)
        self.conv2 = torch.nn.Conv1d(in_channels = self.nlatent,    out_channels = self.nlatent//2, kernel_size = 1)
        self.conv3 = torch.nn.Conv1d(in_channels = self.nlatent//2, out_channels = self.nlatent//4, kernel_size = 1)
        self.conv4 = torch.nn.Conv1d(in_channels = self.nlatent//4, out_channels = coord_dim,       kernel_size = 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.nlatent)
        self.bn2 = torch.nn.BatchNorm1d(self.nlatent//2)
        self.bn3 = torch.nn.BatchNorm1d(self.nlatent//4)

    

    def forward(self, pgon_embeds):
        '''
        Args:
            pgon_embeds: tensor, shape (batch_size, pgon_embed_dim)
        Return:
            
            polygons:  shape (batch_size, num_vert, coord_dim = 2) 
            rand_grid: shape (batch_size, 2, num_vert)

        '''
        device = pgon_embeds.device
        batch_size, pgon_embed_dim = pgon_embeds.shape

        # Concat grids and pgon_embeds: Bx(C+2)xN
        # pgon_embeds_dup: shape (batch_size, pgon_embed_dim, num_vert)
        pgon_embeds_dup = pgon_embeds.unsqueeze(2).expand(
            batch_size, pgon_embed_dim, self.num_vert).contiguous() # BxCxN

        # generate rand grids
        # rand_grid: shape (batch_size, 2, num_vert)
        rand_grid = generate_rand_grid(self.pgon_dec_grid_init, 
                                    batch_size = batch_size, 
                                    num_vert = self.num_vert, 
                                    extent = self.extent, 
                                    grid_dim = self.grid_dim)
        if self.pgon_dec_grid_enc_type == "none":
            rand_grid = torch.FloatTensor(rand_grid).to(device)
        elif self.pgon_dec_grid_enc_type == "spa_enc":
            # rand_grid: np.array(), shape (batch_size, num_vert, 2)
            rand_grid = rand_grid.transpose(0,2,1)
            # rand_grid: torch.tensor(), shape (batch_size, num_vert, spa_embed_dim)
            rand_grid = self.spa_enc(rand_grid)
            # rand_grid: torch.tensor(), shape (batch_size, spa_embed_dim, num_vert)
            rand_grid = rand_grid.permute(0,2,1)
            
        else:
            raise Exception("Unknown pgon_dec_grid_enc_type")
        

        
        # x: shape (batch_size, 2 + pgon_embed_dim, num_vert) or (batch_size, spa_embed_dim + pgon_embed_dim, num_vert)
        x = torch.cat([rand_grid, pgon_embeds_dup], dim=1)

        # Generate points
        # x: shape (batch_size, 2 + pgon_embed_dim, num_vert)
        x = F.relu(self.bn1(self.conv1(x)))

        # x: shape (batch_size, (2 + pgon_embed_dim)/2, num_vert)
        x = F.relu(self.bn2(self.conv2(x)))

        # x: shape (batch_size, (2 + pgon_embed_dim)/4, num_vert)
        x = F.relu(self.bn3(self.conv3(x)))

        # x: shape (batch_size, coord_dim, num_vert)
        x = self.th(self.conv4(x))

        # polygons:  shape (batch_size, num_vert, coord_dim = 2) 
        polygons = x.transpose(2, 1)

        # rand_grid_:  shape (batch_size, num_vert, grid_dim = 2) 
        rand_grid_ = rand_grid.transpose(2, 1)
        return polygons, rand_grid_





class ExplicitConvPolygonDecoder(nn.Module):
    """
    
    """
    def __init__(self, spa_enc, pgon_embed_dim, num_vert, pgon_dec_grid_init = "uniform", pgon_dec_grid_enc_type = "none",
                coord_dim = 2, padding_mode = 'circular', extent = (-1, 1, -1, 1), device = "cpu"):
        """
        Args:
            spa_enc: a spatial encoder
            pgon_embed_dim: the output polygon embedding dimention
            num_vert: number of uniuqe vertices each generated polygon with have
                note that this does not include the extra point (same as the 1st one) to close the ring
            pgon_dec_grid_init: We generate a list of grid points for polygon decoder, the type of grid points are:
                uniform: points uniformly sampled from (-1, 1, -1, 1) 
                circle: points sampled equal-distance on a circle whose radius is randomly sampled
                kdgrid: k-d regular grid, (num_vert - k^2) is uniformly sampled
            pgon_dec_grid_enc_type: the type to encode the grid point
                none: no encoding, use the original grid point
                spa_enc: use space encoder to encode grid point before 
            coord_dim: 2
            padding_mode: 'circular'
            device:
        """
        super(ExplicitConvPolygonDecoder, self).__init__()
        
        self.spa_enc = spa_enc
        self.spa_embed_dim = self.spa_enc.spa_embed_dim
        self.pgon_embed_dim = pgon_embed_dim
        self.num_vert = num_vert
        self.pgon_dec_grid_init = pgon_dec_grid_init
        self.coord_dim = coord_dim
        self.extent = extent

        self.device = device
        self.grid_dim = 2

        self.pgon_dec_grid_enc_type = pgon_dec_grid_enc_type

        # Partly borrowed from atlasnetV2
        # define point generator 
        if self.pgon_dec_grid_enc_type == "none":
            self.nlatent = self.pgon_embed_dim + self.grid_dim
        elif self.pgon_dec_grid_enc_type == "spa_enc":
            self.nlatent = self.pgon_embed_dim + self.spa_embed_dim
        else:
            raise Exception("Unknown pgon_dec_grid_enc_type")

        # by default the bias = True, so this is like a individual MLP
        self.conv1 = torch.nn.Conv1d(in_channels = self.nlatent,    out_channels = self.nlatent,    kernel_size = 3, 
                                    stride=1, padding=1, padding_mode = padding_mode)
        self.conv2 = torch.nn.Conv1d(in_channels = self.nlatent,    out_channels = self.nlatent//2, kernel_size = 3, 
                                    stride=1, padding=1, padding_mode = padding_mode)
        self.conv3 = torch.nn.Conv1d(in_channels = self.nlatent//2, out_channels = self.nlatent//4, kernel_size = 3, 
                                    stride=1, padding=1, padding_mode = padding_mode)
        self.conv4 = torch.nn.Conv1d(in_channels = self.nlatent//4, out_channels = coord_dim,       kernel_size = 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.nlatent)
        self.bn2 = torch.nn.BatchNorm1d(self.nlatent//2)
        self.bn3 = torch.nn.BatchNorm1d(self.nlatent//4)

    

    def forward(self, pgon_embeds):
        '''
        Args:
            pgon_embeds: tensor, shape (batch_size, pgon_embed_dim)
        Return:
            
            polygons:  shape (batch_size, num_vert, coord_dim = 2) 
            rand_grid: shape (batch_size, 2, num_vert)

        '''
        device = pgon_embeds.device
        batch_size, pgon_embed_dim = pgon_embeds.shape

        # Concat grids and pgon_embeds: Bx(C+2)xN
        # pgon_embeds_dup: shape (batch_size, pgon_embed_dim, num_vert)
        pgon_embeds_dup = pgon_embeds.unsqueeze(2).expand(
            batch_size, pgon_embed_dim, self.num_vert).contiguous() # BxCxN

        # generate rand grids
        # rand_grid: shape (batch_size, 2, num_vert)
        rand_grid = generate_rand_grid(self.pgon_dec_grid_init, 
                                    batch_size = batch_size, 
                                    num_vert = self.num_vert, 
                                    extent = self.extent, 
                                    grid_dim = self.grid_dim)
        if self.pgon_dec_grid_enc_type == "none":
            rand_grid = torch.FloatTensor(rand_grid).to(device)
        elif self.pgon_dec_grid_enc_type == "spa_enc":
            # rand_grid: np.array(), shape (batch_size, num_vert, 2)
            rand_grid = rand_grid.transpose(0,2,1)
            # rand_grid: torch.tensor(), shape (batch_size, num_vert, spa_embed_dim)
            rand_grid = self.spa_enc(rand_grid)
            # rand_grid: torch.tensor(), shape (batch_size, spa_embed_dim, num_vert)
            rand_grid = rand_grid.permute(0,2,1)
            
        else:
            raise Exception("Unknown pgon_dec_grid_enc_type")
        

        
        # x: shape (batch_size, 2 + pgon_embed_dim, num_vert) or (batch_size, spa_embed_dim + pgon_embed_dim, num_vert)
        x = torch.cat([rand_grid, pgon_embeds_dup], dim=1)

        # Generate points
        # x: shape (batch_size, 2 + pgon_embed_dim, num_vert)
        x = F.relu(self.bn1(self.conv1(x)))

        # x: shape (batch_size, (2 + pgon_embed_dim)/2, num_vert)
        x = F.relu(self.bn2(self.conv2(x)))

        # x: shape (batch_size, (2 + pgon_embed_dim)/4, num_vert)
        x = F.relu(self.bn3(self.conv3(x)))

        # x: shape (batch_size, coord_dim, num_vert)
        x = self.th(self.conv4(x))

        # polygons:  shape (batch_size, num_vert, coord_dim = 2) 
        polygons = x.transpose(2, 1)

        # rand_grid_:  shape (batch_size, num_vert, grid_dim = 2) 
        rand_grid_ = rand_grid.transpose(2, 1)
        return polygons, rand_grid_