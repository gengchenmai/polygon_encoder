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


from polygonembed.dataset import * 



def json_load(filepath):
    with open(filepath, "r") as json_file:
        data = json.load(json_file)
    return data

def json_dump(data, filepath, pretty_format = True):
    with open(filepath, 'w') as fw:
        if pretty_format:
            json.dump(data, fw, indent=2, sort_keys=True)
        else:
            json.dump(data, fw)

def pickle_dump(obj, pickle_filepath):
    with open(pickle_filepath, "wb") as f:
        pickle.dump(obj, f, protocol=2)

def pickle_load(pickle_filepath):
    with open(pickle_filepath, "rb") as f:
        obj = pickle.load(f)
    return obj



# ########################## GeoData Utility

def make_projected_gdf(gdf, geometry_col = "geometry", epsg_code = 4326):
    gdf[geometry_col] = gdf[geometry_col].to_crs(epsg=epsg_code)
    gdf.crs = from_epsg(epsg_code)
    return gdf

def explode(indata):
    if type(indata) == gpd.GeoDataFrame:
        indf = indata
    elif type(indata) == str:
        indf = gpd.GeoDataFrame.from_file(indata)
    else:
        raise Exception("Input no recognized")
    outdf = gpd.GeoDataFrame(columns=indf.columns, crs = indf.crs)
    for idx, row in indf.iterrows():
        if type(row.geometry) == Polygon:
            outdf = outdf.append(row,ignore_index=True)
        if type(row.geometry) == MultiPolygon:
            multdf = gpd.GeoDataFrame(columns=indf.columns)
            recs = len(row.geometry)
            multdf = multdf.append([row]*recs,ignore_index=True)
            for geom in range(recs):
                multdf.loc[geom,'geometry'] = row.geometry[geom]
            outdf = outdf.append(multdf,ignore_index=True)
    return outdf

def plot_multipolygon(multipygon, figsize = (32, 32), edgecolor='r', ax = None):
    if ax is None:
        fig, ax = plt.subplots(figsize = figsize)
    p = gpd.GeoDataFrame(geometry = [multipygon])
    p.geometry.boundary.plot(edgecolor=edgecolor, ax = ax)
    if ax is None:
        plt.show()

def get_bbox_of_df(indf):
    bbox_df = indf["geometry"].bounds

    minx = np.min(bbox_df["minx"])
    miny = np.min(bbox_df["miny"])
    maxx = np.max(bbox_df["maxx"])
    maxy = np.max(bbox_df["maxy"])
    extent = (minx, miny, maxx, maxy)
    return bbox_df, extent


def get_polygon_exterior_max_num_vert(indf, col = "geometry"):
    coord_len_list = list(indf[col].apply(lambda x: len(x.exterior.coords) ))
    return max(coord_len_list)


def upsample_polygon_exterior_gdf_by_num_vert(indf, num_vert):
    return indf["geometry"].apply(
        lambda x: Polygon(line_interpolate_by_num_vert(x.exterior, num_vert = num_vert).coords ) )

def line_interpolate_by_num_vert(geom, num_vert):
    if geom.geom_type in ['LineString', 'LinearRing']:
        num_vert_origin = len(geom.coords.xy[0])

        if num_vert_origin == num_vert:
            return geom
        elif num_vert_origin > num_vert:
            raise Exception("The original line has larger number of vetice then your input")
        num_vert_add = num_vert - num_vert_origin

        # print(num_vert, num_vert_origin, num_vert_add)
        pt_add_list = []
        dist_add_list = []
        for i in range(1, num_vert_add+1):
            pt_add = geom.interpolate(float(i) / (num_vert_add+1), normalized=True)
            dist_add = geom.project(pt_add)

            pt_add_list.append(pt_add)
            dist_add_list.append(dist_add)

        for idx in range(1, num_vert_origin - 1):
            pt = Point(geom.coords[idx])
            dist = geom.project(pt)
            insert_idx = np.searchsorted(dist_add_list, dist)

            dist_add_list = dist_add_list[:insert_idx] + [dist] + dist_add_list[insert_idx:]
            pt_add_list   = pt_add_list[:insert_idx]   + [pt]   + pt_add_list[insert_idx:]


        pt_add_list = [Point(geom.coords[0])] + pt_add_list + [Point(geom.coords[0])]
        if geom.geom_type == 'LineString':
            return LineString(pt_add_list)
        elif geom.geom_type == 'LinearRing':
            line = LineString(pt_add_list)
            return LinearRing(line.coords)
    else:
        raise ValueError('unhandled geometry %s', (geom.geom_type,))

def normalize_geometry_by_extent(geom, extent = None):
    '''
    Normalize the polygon coords to x: (-1, 1), y: (-1, 1)
    Args:
        geom: a geometry
        extent: (minx, miny, maxx, maxy)
    '''
    if extent is not None:
        minx, miny, maxx, maxy = extent
    else:
        minx, miny, maxx, maxy = geom.bounds

    assert minx < maxx
    assert miny < maxy
    # compute extent center
    x_c = (maxx + minx)/2
    y_c = (maxy + miny)/2
    # 1. affinity to the extent's center
    geom_aff = shapely.affinity.affine_transform(geom, matrix = [1, 0, 0, 1, -x_c, -y_c])
    # plot_multipolygon(geom_aff, figsize =  (5, 5))
    
    
    deltax = maxx - minx
    deltay = maxy - miny
    if deltax >= deltay:
        max_len = deltax
    else:
        max_len = deltay
    # 2. scale to x: (-1, 1), y: (-1, 1)
    geom_scale = shapely.affinity.scale(geom_aff, xfact=2.0/max_len, yfact=2.0/max_len, zfact=0, origin='center')
    # plot_multipolygon(geom_scale, figsize =  (5, 5))
    return geom_scale



def load_dataframe(data_dir, filename):
    if filename.endswith(".pkl"):
        ingdf = pickle_load(os.path.join(data_dir, filename))
    else:
        raise Exception('Unknow file type')
    return ingdf

    