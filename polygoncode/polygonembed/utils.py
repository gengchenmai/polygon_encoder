
import numpy as np
import torch
import json
import os
import math
import pickle
import logging
import random
import time

import shapely
from shapely.ops import transform

from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.linestring import LineString
from shapely.geometry.polygon import LinearRing
from shapely.geometry import box

import geopandas as gpd



def get_perfect_nth_root(num, power):
    candidate = num ** (1/power)
    low = int(math.floor(candidate))
    high = int(math.ceil(candidate))
    if num == low**power:
        return low
    elif num == high**power:
        return high
    else:
        return low


def scale_max_and_min_radius(geom_type, extent, max_radius, min_radius):
    if geom_type == "norm":
        return max_radius, min_radius
    elif geom_type == "origin":
        minx, maxx, miny, maxy = extent
        deltax = maxx - minx
        deltay = maxy - miny
        if deltax >= deltay:
            max_len = deltax
        else:
            max_len = deltay
        scale = max_len/2.0
        return max_radius*scale, min_radius*scale


def generate_rand_grid(pgon_dec_grid_init, batch_size, num_vert, extent, grid_dim = 2):
    '''
    generate rand grid points as https://arxiv.org/pdf/1712.07262.pdf
    Args:
        pgon_dec_grid_init: We generate a list of grid points for polygon decoder, the type of grid points are:
                    uniform: points uniformly sampled from (-1, 1, -1, 1) 
                    circle: points sampled equal-distance on a circle whose radius is randomly sampled
                    2dgrid: 2d regular grid
        batch_size:
        grid_dim: the dimention of points 
        num_vert: numerb of vertices/points to be generated
        extent: (-1, 1, -1, 1),
        
    Return:
        rand_grid: np.array(), points in the extent, shape (batch_size, grid_dim = 2, num_vert)
    '''

    # we first generate grid points into (-1, 1, -1, 1), if extent is not that, we do a shift
    start, end = (-1, 1)
    if pgon_dec_grid_init == "uniform":
        # rand_grid: shape (batch_size, grid_dim = 2, num_vert)
        # rand_grid = torch.FloatTensor(
        #     batch_size, grid_dim, num_vert).to(device) # Bx2xN
        # rand_grid.data.uniform_(start, end)
        rand_grid = np.random.uniform(start, end, size = (batch_size, grid_dim, num_vert))
    elif pgon_dec_grid_init == "kdgrid":
        # number of point per dimention
        num_pt_per_dim = get_perfect_nth_root(num_vert, grid_dim)
        # some extra points
        num_vert_rest = num_vert - num_pt_per_dim**grid_dim
        grid_pt_list = []
        for i in range(grid_dim):
            grid_pt_list.append( np.linspace(start, end, num=num_pt_per_dim) )
        # kd_grids: shape (grid_dim, num_pt_per_dim**grid_dim)
        kd_grids = np.array(np.meshgrid(*grid_pt_list)).reshape(grid_dim, -1)
        # kd_grids: shape (1, grid_dim, num_pt_per_dim**grid_dim)
        kd_grids = np.expand_dims(kd_grids, axis = 0)
        # kd_grids: shape (batch_size, grid_dim, num_pt_per_dim**grid_dim)
        kd_grids = np.repeat(kd_grids, repeats = batch_size, axis = 0)
        # rand_grid: shape (batch_size, grid_dim, num_pt_per_dim**grid_dim)
        rand_grid = kd_grids
        if num_vert_rest > 0:
            # uniformly sample the rest points
            rand_grid_rest = np.random.uniform(start, end, size = (batch_size, grid_dim, num_vert_rest))
            
            rand_grid = np.concatenate([rand_grid, rand_grid_rest], axis = 2)
    elif pgon_dec_grid_init == "circle":
        assert grid_dim == 2
        # make radius of each ring
        radius = end - (start + end)/2
        # radiuss: (batch_size), the radius of each ring
        radiuss = np.random.power(3, batch_size)*radius
        # radiuss: (batch_size, num_vert), the radius of each ring
        radiuss_ = np.repeat(np.expand_dims(radiuss, axis = 1), repeats = num_vert, axis = 1)

        # make theta offset for each ring
        theta_interval = np.pi/num_vert
        # theta_offsets: (batch_size), the theta offset of the 1st interval of each ring
        theta_offsets = np.random.uniform(0, theta_interval, size = batch_size)
        # theta_offsets: (batch_size, num_vert), the theta offset of the 1st interval of each ring
        theta_offsets = np.repeat(np.expand_dims(theta_offsets, axis = 1), repeats = num_vert, axis = 1)
        # make theta for each ring
        thetas = np.linspace(0, 2*np.pi, num = num_vert, endpoint = False)
        # thetas: shape (batch_size, num_vert)
        thetas = np.repeat(np.expand_dims(thetas, axis = 0), repeats = batch_size, axis = 0)
        # thetas: shape (batch_size, num_vert), the theta for each ring vert
        thetas_ = thetas + theta_offsets

        # make x,y  coordinate
        # x, y: shape (batch_size, num_vert)
        x = radiuss_ * np.cos(thetas_)
        y = radiuss_ * np.sin(thetas_)

        rand_grid = np.zeros((batch_size, grid_dim, num_vert))
        rand_grid[:, 0, :] = x
        rand_grid[:, 1, :] = y
        # rand_grid = torch.FloatTensor(rand_grid).to(device)
    
    if extent != (-1, 1, -1, 1):
        rand_grid = affinity_pts_by_extent(rand_grid, extent = extent)
    return rand_grid


def affinity_pts_by_extent(pts, extent = (-1, 1, -1, 1)):
    '''
    Normalize the pt coords from (-1, 1, -1, 1) to extent
    Args:
        pts: shape (batch_size, grid_dim = 2, num_vert)
        extent: (minx, miny, maxx, maxy)
    '''
    assert np.sum(rand_grid[:, 0, :] < -1) == 0
    assert np.sum(rand_grid[:, 0, :] >  1) == 0
    assert np.sum(rand_grid[:, 1, :] < -1) == 0
    assert np.sum(rand_grid[:, 1, :] >  1) == 0

    assert extent[0] < extent[1]
    assert extent[2] < extent[3]
    minx, maxx, miny, maxy = extent
    
    pts_ = copy.deepcopy(pts)
    # 1. scale to x: (minx, maxx), y: (miny, maxy)
    deltax = maxx - minx
    deltay = maxy - miny
    if deltax >= deltay:
        max_len = deltax
    else:
        max_len = deltay
    
    pts_[:, 0, :] = pts_[:, 0, :] * max_len/2.0
    pts_[:, 1, :] = pts_[:, 1, :] * max_len/2.0
    
    
    # compute extent center
    x_c = (maxx + minx)/2
    y_c = (maxy + miny)/2
    # 2. affinity to the extent's center
    pts_[:, 0, :] = pts_[:, 0, :] + x_c
    pts_[:, 1, :] = pts_[:, 1, :] + y_c
    
    return pts_


def make_polygons_from_coords(polygons_seq):
    '''
    Args:
        polygons_seq: shape (batch_size, num_vert, coord_dim = 2)
    '''
    batch_size, num_vert, coord_dim = polygons_seq.shape
    polygons = []
    for i in range(batch_size):
        line = LineString([ [polygons_seq[i, j, 0], polygons_seq[i, j, 1]] for j in range(num_vert) ])
        polygon = Polygon(line.coords)
        polygons.append(polygon)
    return polygons


def random_start_polygon_coords_with_last_pt(polygons):
    '''
    Randomly pick a index as the new start of the polygon exterior to start this polygon coords sequence
    Args:
        polygons: np.array shape (batch_size, num_vert+1, coord_dim = 2) 
                note that in num_vert dimention, the last point is the same as the 1st point
    '''
    
    polygons_ = polygons[:, :-1, :]
    batch_size, num_vert, coord_dim = polygons.shape
    start_idx = np.random.choice(np.arange(0, num_vert))
    if type(polygons)  == torch.Tensor :
        end_verts = torch.unsqueeze(polygons_[:, start_idx, :], dim = 1)
        polygons_random = torch.cat([polygons_[:, start_idx:, :], polygons_[:, :start_idx, :], end_verts], dim = 1)
    elif type(polygons)  == np.ndarray :
        end_verts = np.expand_dims(polygons_[:, start_idx, :], axis = 1)
        polygons_random = np.concatenate([polygons_[:, start_idx:, :], polygons_[:, :start_idx, :], end_verts], axis = 1)
    else:
        raise Except("Unknown polygons tensor type")
    return polygons_random

def random_start_polygon_coords(polygons):
    '''
    Randomly pick a index as the new start of the polygon exterior to start this polygon coords sequence
    Args:
        polygons: np.array, shape (batch_size, num_vert, coord_dim = 2) 
                here we assume each vert in unique
    '''
    batch_size, num_vert, coord_dim = polygons.shape
    start_idx = np.random.choice(np.arange(0, num_vert))
    if type(polygons)  == torch.Tensor :
        polygons_random = torch.cat([polygons[:, start_idx:, :], polygons[:, :start_idx, :]], dim = 1)
    elif type(polygons)  == np.ndarray :
        polygons_random = np.concatenate([polygons[:, start_idx:, :], polygons[:, :start_idx, :]], axis = 1)
    else:
        raise Except("Unknown polygons tensor type")
    return polygons_random


def flip_polygons(polygons, device = "cpu"):
    '''
    Args:
        polygons: shape (batch_size, 2, num_vert)
    Return:
        flip_polygons: shape (batch_size, 2, num_vert), flip all polygon upside down
    '''
    flip_matrix = [[-1, 0],[0,1]]

    if type(polygons)  == torch.Tensor :
        flip_matrix = torch.FloatTensor(flip_matrix).to(device)
        flip_pgons = torch.einsum('kc,bcl->bkl', flip_matrix, polygons)
    elif type(polygons)  == np.ndarray :
        flip_pgons = np.einsum('kc,bcl->bkl', flip_matrix, polygons)
    return flip_pgons

def rotate_polygons(polygons, theta, device = "cpu"):
    '''
    Args:
        polygons: shape (batch_size, 2, num_vert)
        theta: rotate angle, in radius
    Return:
        rotate_polygons: shape (batch_size, 2, num_vert), rotate all polygon by theta
    '''
    '''
    rotation matrix:
        [[cos𝜃, -sin𝜃],
         [sin𝜃,  cos𝜃]]
    '''
    rotate_matrix = [[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]]

    if type(polygons)  == torch.Tensor :
        rotate_matrix = torch.FloatTensor(rotate_matrix).to(device)
        rotate_pgons = torch.einsum('kc,bcl->bkl', rotate_matrix, polygons)
    elif type(polygons)  == np.ndarray :
        rotate_pgons = np.einsum('kc,bcl->bkl', rotate_matrix, polygons)
    return rotate_pgons

def translate_polygons(polygons, delta = 0.1, device = "cpu"):
    '''
    Args:
        polygons: shape (batch_size, num_vert, 2)
    Return:
        flip_polygons: shape (batch_size, num_vert, 2), translate all polygon with (deltaX, deltaY)
    '''
    deltaXY = np.random.uniform(low=-delta, high=delta, size=(2))
    if type(polygons)  == torch.Tensor :
        deltaXY = torch.FloatTensor(deltaXY).to(device)
        polygons_translate = polygons + deltaXY
    elif type(polygons)  == np.ndarray :
        polygons_translate = polygons + deltaXY
    return polygons_translate

def scale_polygons(polygons, min_scale = 0.9, max_scale = 1.1):
    '''
    Args:
        polygons: shape (batch_size, num_vert, 2)
    Return:
        scale_polygons: shape (batch_size, num_vert, 2), translate all polygon with (deltaX, deltaY)
    '''
    scale = np.random.uniform(low=min_scale, high=max_scale)
    return polygons * scale

def noise_polygons(polygons, delta = 0.01, device = "cpu"):
    '''
    Args:
        polygons: shape (batch_size, num_vert, 2) or (batch_size, 2, num_vert)
    Return:
        polygons_noise: shape (batch_size, num_vert, 2) or (batch_size, 2, num_vert), 
                add white noise to each polygon vertices
    '''
    noise = np.random.uniform(low=-delta, high=delta, size = polygons.shape)
    if type(polygons)  == torch.Tensor :
        noise = torch.FloatTensor(noise).to(device)
        polygons_noise = polygons + noise
    elif type(polygons)  == np.ndarray :
        polygons_noise = polygons + noise
    return polygons_noise

def polygon_data_augment(polygons, data_augment_type, device = "cpu"):
    '''
    do data augmentation for each polygon
    Args:
        polygons: shape (batch_size,  num_vert, 2), can be the simple polygon vertices, or the V matrix of complex polygons
    Return:
        polygons_aug: shape (batch_size * N,  num_vert, 2), N is decided based on data_augment_type
    '''
    polygons_list = [polygons]

    # polygons_: shape (batch_size,  2, num_vert)
    polygons_ = polygons.permute(0, 2, 1)
    if "flp" in data_augment_type:
        polygons_flip = flip_polygons(polygons_, device)
        polygons_list.append(polygons_flip.permute(0, 2, 1))
    if "rot" in data_augment_type:
        theta = np.random.uniform(-10, 10)*np.pi/180
        polygons_rotate = rotate_polygons(polygons_, theta, device)
        polygons_list.append(polygons_rotate.permute(0, 2, 1))
    if "tra" in data_augment_type:
        # polygons_translate = translate_polygons(polygons, delta = 0.1, device = device)
        polygons_translate = translate_polygons(polygons, delta = 0.01, device = device)
        polygons_list.append(polygons_translate)
    if "scl" in data_augment_type:
        # polygons_scale = scale_polygons(polygons, min_scale = 0.9, max_scale = 1.1)
        polygons_scale = scale_polygons(polygons, min_scale = 0.9, max_scale = 1.0)
        polygons_list.append(polygons_scale)
    if "noi" in data_augment_type:
        # polygons_noise = noise_polygons(polygons, delta = 0.01, device = device)
        polygons_noise = noise_polygons(polygons, delta = 0.001, device = device)
        polygons_list.append(polygons_noise)
        
    polygons_aug = torch.cat(polygons_list, dim = 0)
    return polygons_aug

def random_flip_rotate_scale_polygons(polygons, flip_flag = None, theta = None, device = "cpu"):
    '''
    flip, rotate, and scale each polygon to do data argumentation
    Args:
        polygons: shape (batch_size,  num_vert, 2)
    Return:
        rotate_pgons: shape (batch_size, num_vert, 2), 
    '''
    polygons_ = polygons.permute(0, 2, 1)
    
    # 1. flip polygon with 0.5 chance
    # flip_flag = np.random.choice([0,1])
    # if flip_flag == 1:
    #     polygons_ = flip_polygons(polygons_, device)
    # 2. rotate polygon with a random angle
    if theta is not None:
        theta = np.random.uniform(0, 2*np.pi)
    polygons_ = rotate_polygons(polygons_, theta, device)
    # # 3. scale polygon with a random scale [0.5, 1]
    # scale = np.random.uniform(0.5, 1)
    # polygons_ = polygons_ * scale
    
    rotate_pgons = polygons_.permute(0, 2, 1)
    return rotate_pgons


def rotate_polygon_batch(pgon_id_list, pgon_list, class_list = None, batch_size = 500, num_augment = 8):
    data_len = pgon_list.shape[0]
    # do polygon rotate
    theta_list = np.arange(num_augment)*2*np.pi/num_augment
    pgon_id_list_rot = []
    pgon_list_rot = []
    class_list_rot = []
    # not include the original polygon
    for theta in theta_list[1:]:
        # theta = theta_list[2]
        pgon_rotate_list = []
        for start in np.arange(0,data_len,step = batch_size):
            end = min(start + batch_size, data_len)
            # print(id_list[start:end])
            polygons = pgon_list[start:end]
            polygons = np.transpose(polygons, (0, 2, 1))
            pgon_rotate = rotate_polygons(polygons, theta)
            pgon_rotate_list.append(pgon_rotate)
        pgon_rotate_list = np.concatenate(pgon_rotate_list, axis = 0)
        pgon_rotate_list = np.transpose(pgon_rotate_list, (0, 2, 1))

        pgon_id_list_rot.append(pgon_id_list)
        pgon_list_rot.append(pgon_rotate_list)
        if class_list is not None:
            class_list_rot.append(class_list)

    pgon_id_list_rot = np.concatenate(pgon_id_list_rot, axis = 0)
    pgon_list_rot = np.concatenate(pgon_list_rot, axis = 0)
    if class_list is not None:
        class_list_rot = np.concatenate(class_list_rot, axis = 0)
    else:
        class_list_rot = None
    return pgon_id_list_rot, pgon_list_rot, class_list_rot

def flip_polygon_batch(pgon_id_list, pgon_list, class_list = None, batch_size = 500):
    data_len = pgon_list.shape[0]
    pgon_flip_list = []
    for start in np.arange(0,data_len,step = batch_size):
        end = min(start + batch_size, data_len)
        # print(id_list[start:end])
        polygons = pgon_list[start:end]
        polygons = np.transpose(polygons, (0, 2, 1))
        pgon_flip = flip_polygons(polygons)
        pgon_flip_list.append(pgon_flip)
    pgon_flip_list = np.concatenate(pgon_flip_list, axis = 0)
    pgon_flip_list = np.transpose(pgon_flip_list, (0, 2, 1))

    return pgon_id_list, pgon_flip_list, class_list

def add_first_pt_to_polygons(polygons_seq):
    '''
    add first point to the polygon sequence
    Args:
        polygons_seq: torch.FolarTensor(), shape (batch_size, num_vert, coord_dim = 2) 
    Return:
        polygons: torch.FolarTensor(), shape (batch_size, num_vert+1, coord_dim = 2) 
    '''
    # first_pts: shape (batch_size, 1, coord_dim = 2) 
    first_pts = polygons_seq[:, 0, :].unsqueeze(1)
    # polygons: shape (batch_size, num_vert+1, coord_dim = 2)
    polygons = torch.cat([polygons_seq, first_pts], dim=1)
    return polygons


def batch_pairwise_dist(pts1, pts2):
    '''
    compute pairewise distance matrix between two point sets
    Args:
        pts1: shape (batch_size, num_vert_1, coord_dim) => (B,M,2)
        pts2: shape (batch_size, num_vert_2, coord_dim) => (B,N,2)
    Return:
        dist: shape (batch_size, num_vert_1, num_vert_2) => (B,M,N)
    '''
    r_pts1 = torch.sum(pts1 * pts1, dim=2, keepdim=True)  # (B,M,1)
    r_pts2 = torch.sum(pts2 * pts2, dim=2, keepdim=True)  # (B,N,1)
    mul = torch.matmul(pts1, pts2.permute(0,2,1))         # (B,M,N)
    dist = r_pts1 - 2 * mul + r_pts2.permute(0,2,1)       # (B,M,N)
    return dist




def batch_nearest_neighbor_loss(x, y):
    '''
    compute nearest neighbor loss between two point sets
    Args:
        x: shape (batch_size, num_vert_1, coord_dim) => (B,M,2)
        y: shape (batch_size, num_vert_2, coord_dim) => (B,N,2)
    Return:
        nn_loss: shape (batch_size)
    '''
    # dist: shape (batch_size, num_vert_1, num_vert_2) => (B,M,N)
    dist = batch_pairwise_dist(x,y)
    # values_1, indices_1: shape (batch_size, num_vert_2) => (B, N)
    values_1, indices_1 = dist.min(dim = 1)
    # min_dist_1: shape (batch_size, 1)
    min_dist_1 = torch.mean(values_1, dim=-1, keepdim = True)

    # values_2, indices_2: shape (batch_size, num_vert_1) => (B, M)
    values_2, indices_2 = dist.min(dim = 2)
    # min_dist_2: shape (batch_size, 1)
    min_dist_2 = torch.mean(values_2, dim=-1, keepdim = True)

    # min_dist: shape (batch_size, 2)
    min_dist = torch.cat([min_dist_1, min_dist_2], dim = -1)
    # nn_loss: shape (batch_size)
    nn_loss, indices = torch.max(min_dist, dim = -1, keepdim=False)
    return nn_loss


def make_loop_l2_mask(num_vert):
    '''
    make a mask to uniquely label each batch's each diagonal
    return:
        maskD: np.array, shape (num_vert, num_vert, num_vert)
    '''
    maskA = np.sum(np.indices( (num_vert, num_vert) ), axis=0)
    maskB = np.where(maskA >= num_vert, maskA - num_vert, maskA)
    '''
    maskC: if num_vert = 5
    [[4, 3, 2, 1, 0],
     [0, 4, 3, 2, 1],
     [1, 0, 4, 3, 2],
     [2, 1, 0, 4, 3],
     [3, 2, 1, 0, 4]]
    '''
    maskC = np.fliplr(maskB)
    # maskC
    maskD = np.repeat( np.expand_dims(maskC, axis = 0), repeats = num_vert, axis = 0)


    labels = np.arange(num_vert)
    # labels: shape (num_vert, 1)
    labels = np.expand_dims(labels, axis = 1)
    # labels: shape (num_vert, num_vert*num_vert)
    labels = np.repeat(labels, repeats = num_vert*num_vert, axis = 1)
    # labels: shape (num_vert, num_vert, num_vert)
    labels = labels.reshape(num_vert, num_vert, num_vert)

    '''
    maskE: shape (num_vert, num_vert, num_vert), sperated diagonal masks
    [[[0, 0, 0, 0, 1],
     [1, 0, 0, 0, 0],
     [0, 1, 0, 0, 0],
     [0, 0, 1, 0, 0],
     [0, 0, 0, 1, 0]],

    [[0, 0, 0, 1, 0],
     [0, 0, 0, 0, 1],
     [1, 0, 0, 0, 0],
     [0, 1, 0, 0, 0],
     [0, 0, 1, 0, 0]],

    [[0, 0, 1, 0, 0],
     [0, 0, 0, 1, 0],
     [0, 0, 0, 0, 1],
     [1, 0, 0, 0, 0],
     [0, 1, 0, 0, 0]],

    [[0, 1, 0, 0, 0],
     [0, 0, 1, 0, 0],
     [0, 0, 0, 1, 0],
     [0, 0, 0, 0, 1],
     [1, 0, 0, 0, 0]],

    [[1, 0, 0, 0, 0],
     [0, 1, 0, 0, 0],
     [0, 0, 1, 0, 0],
     [0, 0, 0, 1, 0],
     [0, 0, 0, 0, 1]]]
    '''
    maskE = maskD == labels
    return maskE


def batch_loop_l2_loss(x, y, loop_l2_mask = None, device = "cpu"):
    '''
    for each polygon pair (P, Q), Given a specific k, where k = 0,1,2,..num_vert-1
    we compute the sum of the distance between P[i] and Q[i+k]
    loop_l2_loss is the min distance between (P, Q)
    Args:
        x: shape (batch_size, num_vert_1, coord_dim) => (B,M,2)
        y: shape (batch_size, num_vert_2, coord_dim) => (B,N,2)
        we assume num_vert_1 == num_vert_2
    Return:
        dist_loopl2_min: shape (batch_size), the min sum of distance between (P, Q)
    '''
    
    # compute pairwise distance matrix
    # dist: shape (batch_size, num_vert_1, num_vert_2) => (B,M,N)
    dist = batch_pairwise_dist(x,y)
    batch_size, num_vert, num_vert_ = dist.shape

    # we only handle polygon pair (P,Q) with the same number of vertices
    assert num_vert == num_vert_

    use_mask = False
    if loop_l2_mask is not None:
        H, M, N = loop_l2_mask.shape
        if H == num_vert and M == num_vert and N == num_vert:
            # print("usa Mask")
            use_mask = True
    if not use_mask:
        # loop_l2_mask: shape (num_vert, num_vert, num_vert)
        loop_l2_mask = make_loop_l2_mask(num_vert)

    mask = torch.LongTensor(loop_l2_mask).to(device)

    '''
    dist_repeat: shape (batch_size, num_vert, num_vert, num_vert), 
    1 => different diagonal group
    2, 3 => distance matric
    '''
    dist_repeat = torch.repeat_interleave(dist.unsqueeze(1), repeats = num_vert, dim = 1)

    # mask_dist: shape (batch_size, num_vert, num_vert, num_vert), 
    #      masked distance for each diagonal group
    mask_dist = torch.einsum('bmnk,mnk->bmnk', dist_repeat, mask)

    '''
    dist_loop: shape (batch_size, num_vert), 
    for each polygon pair (P, Q), Given a specific k, where k = 0,1,2,..num_vert-1
    we compute the sum of the distance between P[i] and Q[i+k] 
    ''' 
    dist_loopl2 = mask_dist.sum(3).sum(2)
    # dist_loopl2_min: shape (batch_size), the min sum of distance between (P, Q)
    dist_loopl2_min, indices = torch.min(dist_loopl2, dim = 1, keepdim = False)
    return dist_loopl2_min


    
# def make_loop_l2_mask(batch_size, num_vert):
#     '''
#     make a mask to uniquely label each batch's each diagonal
#     return:
#         maskD: np.array, shape (batch_size, num_vert, num_vert)
#     '''
#     # make a mask for each diagonal elements - maskD
#     maskA = np.sum(np.indices( (num_vert, num_vert) ), axis=0)
#     maskB = np.where(maskA >= num_vert, maskA - num_vert, maskA)
#     '''
#     maskC: if num_vert = 5
#     [[4, 3, 2, 1, 0],
#      [0, 4, 3, 2, 1],
#      [1, 0, 4, 3, 2],
#      [2, 1, 0, 4, 3],
#      [3, 2, 1, 0, 4]]
#     '''
#     maskC = np.fliplr(maskB)


#     # offset to make each batch item (polygon) has unique diagonal label
#     # offset: shape (batch_size)
#     offset = np.linspace(start = 0, stop = (batch_size-1)*num_vert, num = batch_size)
#     # offset: shape (batch_size, 1)
#     offset = np.expand_dims(offset, axis = 1)
#     # offset: shape (batch_size, num_vert*num_vert)
#     offset = np.repeat(offset, repeats = num_vert*num_vert, axis = 1)
#     # offset: shape (batch_size, num_vert, num_vert)
#     offset = offset.reshape(batch_size, num_vert, num_vert)
#     '''
#     maskD: if num_vert = 5, batch_size = 2
#     [[[ 4.,  3.,  2.,  1.,  0.],
#       [ 0.,  4.,  3.,  2.,  1.],
#       [ 1.,  0.,  4.,  3.,  2.],
#       [ 2.,  1.,  0.,  4.,  3.],
#       [ 3.,  2.,  1.,  0.,  4.]],

#      [[ 9.,  8.,  7.,  6.,  5.],
#       [ 5.,  9.,  8.,  7.,  6.],
#       [ 6.,  5.,  9.,  8.,  7.],
#       [ 7.,  6.,  5.,  9.,  8.],
#       [ 8.,  7.,  6.,  5.,  9.]]]
#     '''
#     maskD = offset+maskC
#     return maskD

# def batch_loop_l2_loss(x, y, maskD = None, device = "cpu"):
#     '''
#     for each polygon pair (P, Q), Given a specific k, where k = 0,1,2,..num_vert-1
#     we compute the sum of the distance between P[i] and Q[i+k]
#     loop_l2_loss is the min distance between (P, Q)
#     Args:
#         x: shape (batch_size, num_vert_1, coord_dim) => (B,M,2)
#         y: shape (batch_size, num_vert_2, coord_dim) => (B,N,2)
#         we assume num_vert_1 == num_vert_2
#     Return:
#         dist_loopl2_min: shape (batch_size), the min sum of distance between (P, Q)
#     '''
    
#     # compute pairwise distance matrix
#     # dist: shape (batch_size, num_vert_1, num_vert_2) => (B,M,N)
#     dist = batch_pairwise_dist(x,y)
#     batch_size, num_vert, num_vert_ = dist.shape

#     # we only handle polygon pair (P,Q) with the same number of vertices
#     assert num_vert == num_vert_

#     use_maskD = False
#     if maskD is not None:
#         B, M, N = maskD.shape
#         if B == batch_size and M == num_vert and M == N:
#             # print("usa MaskD")
#             use_maskD = True
#     if not use_maskD:
#         # maskD: shape (batch_size, num_vert, num_vert)
#         maskD = make_loop_l2_mask(batch_size, num_vert)

#     # maskE: shape (batch_size * num_vert * num_vert)
#     maskE = torch.LongTensor(maskD.reshape(-1)).to(device)

#     '''
#     dist_loop: shape (batch_size, num_vert), 
#     for each polygon pair (P, Q), Given a specific k, where k = 0,1,2,..num_vert-1
#     we compute the sum of the distance between P[i] and Q[i+k] 
#     ''' 
#     dist_loopl2 = torch.bincount(maskE, dist.reshape(-1)).reshape(batch_size, num_vert)
#     # dist_loopl2_min: shape (batch_size), the min sum of distance between (P, Q)
#     dist_loopl2_min, indices = torch.min(dist_loopl2, dim = 1, keepdim = False)
#     return dist_loopl2_min



def count_part(geom):
    parts = 0
    if isinstance(geom, shapely.geometry.polygon.Polygon):
        parts = 1
    elif isinstance(geom, shapely.geometry.multipolygon.MultiPolygon):
        parts = len(geom.geoms)
    return parts

def count_holes_for_single_polygon(geom):
    assert isinstance(geom, shapely.geometry.polygon.Polygon)
    return len(geom.interiors)

def count_holes(geom):
    num_holes = 0
    if isinstance(geom, shapely.geometry.polygon.Polygon):
        num_holes = count_holes_for_single_polygon(geom)
    elif isinstance(geom, shapely.geometry.multipolygon.MultiPolygon):
        for pgon in geom.geoms:
            num_holes += count_holes_for_single_polygon(pgon)
    return num_holes

def count_vert_for_single_polygon(geom):
    assert isinstance(geom, shapely.geometry.polygon.Polygon)
    num_vert = len(geom.exterior.xy[0])-1
    for inter in geom.interiors:
        num_vert += len(inter.xy[0])-1
    return num_vert

def count_vert(geom):
    num_vert = 0
    if isinstance(geom, shapely.geometry.polygon.Polygon):
        num_vert = count_vert_for_single_polygon(geom)
    elif isinstance(geom, shapely.geometry.multipolygon.MultiPolygon):
        for pgon in geom.geoms:
            num_vert += count_vert_for_single_polygon(pgon)
    return num_vert



def get_common_bbox(geom_s, geom_o):
    s_minx, s_miny, s_maxx, s_maxy = geom_s.bounds
    o_minx, o_miny, o_maxx, o_maxy = geom_o.bounds

    minx = min(s_minx, o_minx)
    miny = min(s_miny, o_miny)
    maxx = max(s_maxx, o_maxx)
    maxy = max(s_maxy, o_maxy)
    return (minx, miny, maxx, maxy)

def plot_geo_triple(triple_geom_df, idx):
    fig, axs = plt.subplots(nrows=1, ncols = 2, figsize = (20, 10))
    s = triple_geom_df["s"].iloc[idx]
    r = triple_geom_df["r"].iloc[idx]
    o = triple_geom_df["o"].iloc[idx]
    print(make_readable_triple(triple = (s,r,o)))

    geom_s = triple_geom_df["geom_s"].iloc[idx]
    geom_o = triple_geom_df["geom_o"].iloc[idx]

    extent = get_common_bbox(geom_s, geom_o)
    bounds = box(*extent)

    geom_s_norm = triple_geom_df["geom_s_norm"].iloc[idx]
    geom_o_norm = triple_geom_df["geom_o_norm"].iloc[idx]

    plot_multipolygon(geom_s, edgecolor = "r", ax =  axs[0])
    plot_multipolygon(geom_o, edgecolor = "b", ax =  axs[0])
    plot_multipolygon(bounds, edgecolor = "g", ax =  axs[0])

    plot_multipolygon(geom_s_norm, edgecolor = "r", ax =  axs[1])
    plot_multipolygon(geom_o_norm, edgecolor = "b", ax =  axs[1])
    
def plot_multipolygon(multipygon, figsize = (32, 32), edgecolor='r', ax = None):
    if ax is None:
        fig, ax = plt.subplots(figsize = figsize)
    p = gpd.GeoDataFrame(geometry = [multipygon])
    p.geometry.boundary.plot(edgecolor=edgecolor, ax = ax)
    if ax is None:
        plt.show()
    
# SPARQL function
PREFIX2IRI = {
        'dbr': 'http://dbpedia.org/resource/',
        'dbo': 'http://dbpedia.org/ontology/',
        'dbp': 'http://dbpedia.org/property/',
        'wd':  'http://www.wikidata.org/entity/',
        'foaf':'http://xmlns.com/foaf/0.1/',
        'owl': 'http://www.w3.org/2002/07/owl#',
        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
        'rdfs':'http://www.w3.org/2000/01/rdf-schema#'
    }

def make_prefixed_iri(iri):
    for prefix in PREFIX2IRI:
        if PREFIX2IRI[prefix] in iri:
            return iri.replace(PREFIX2IRI[prefix], prefix + ':')
    return iri

def make_readable_triple(triple):
    s, p, o = triple
    s_ = make_prefixed_iri(s)
    p_ = make_prefixed_iri(p)
    o_ = make_prefixed_iri(o)
    return (s_, p_, o_)

def make_readable_triples(triples):
    tri = []
    for triple in triples:
        tri.append(make_readable_triple(triple))
    return tri

def get_connected_triples(iri, triples):
    triple_set = set()
    for idx, (s, p, o) in enumerate(triples):
        if s == iri or p == iri or o == iri:
            triple_set.add( (s, p, o) )
    return list(triple_set)