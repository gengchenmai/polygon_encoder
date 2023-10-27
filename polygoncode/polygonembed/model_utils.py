
import numpy as np
import torch
import json
import os
import math
import pickle
import logging
import random
import time



from torch.utils.data.sampler import Sampler

from polygonembed.atten import *
from polygonembed.module import *
from polygonembed.SpatialRelationEncoder import *
from polygonembed.PolygonEncoder import *
from polygonembed.PolygonDecoder import *
from polygonembed.resnet2d import *
from polygonembed.lenet import *
from polygonembed.enc_dec import *




def setup_console():
    logging.getLogger('').handlers = []
    console = logging.StreamHandler()
    # optional, set the logging level
    console.setLevel(logging.INFO)
    # set a format which is the same for console use
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    
def setup_logging(log_file, console=True, filemode='w'):
    #logging.getLogger('').handlers = []
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=log_file,
                        filemode=filemode)
    if console:
        #logging.getLogger('').handlers = []
        console = logging.StreamHandler()
        # optional, set the logging level
        console.setLevel(logging.INFO)
        # set a format which is the same for console use
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)
    return logging

def get_ffn(args, input_dim, output_dim, f_act, context_str = ""):
    if "ffn_type" in args and "ffn_hidden_layers" in args:
        if args.ffn_type == "ffn":
            return MultiLayerFeedForwardNN(
                input_dim = input_dim,
                output_dim = output_dim,
                num_hidden_layers = args.num_hidden_layer,
                dropout_rate = args.dropout,
                hidden_dim = args.hidden_dim,
                activation = f_act,
                use_layernormalize = args.use_layn,
                skip_connection = args.skip_connection,
                context_str = context_str)
        elif args.ffn_type == "ffnf":
            return MultiLayerFeedForwardNNFlexible(
                input_dim = input_dim,
                output_dim = output_dim,
                hidden_layers = args.ffn_hidden_layers,
                dropout_rate = args.dropout,
                activation = f_act,
                use_layernormalize = args.use_layn,
                skip_connection = args.skip_connection,
                context_str = context_str)
        else:
            raise Exception(f"ffn_type: {args.ffn_type} is not defined")
    else:
        return MultiLayerFeedForwardNN(
                input_dim = input_dim,
                output_dim = output_dim,
                num_hidden_layers = args.num_hidden_layer,
                dropout_rate = args.dropout,
                hidden_dim = args.hidden_dim,
                activation = f_act,
                use_layernormalize = args.use_layn,
                skip_connection = args.skip_connection,
                context_str = context_str)

def get_extent_by_geom_type(pgon_gdf, geom_type):
    # extent: (x_min, x_max, y_min, y_max)
    if geom_type == "norm":
        return (-1, 1, -1, 1)
    elif geom_type == "origin":
        minx, miny, maxx, maxy = list(pgon_gdf.total_bounds)
        return (minx, maxx, miny, maxy)


def get_spa_enc_input_dim(spa_enc_type, frequency_num = 16, coord_dim = 2, num_rbf_anchor_pts = 100, k_delta = 1):
    if spa_enc_type == "gridcell":
        in_dim = int(4 * frequency_num)
    elif spa_enc_type == "gridcellnorm":
        in_dim = int(4 * frequency_num)
    elif spa_enc_type == "hexagridcell":
        in_dim = None
    elif spa_enc_type == "theory":
        in_dim = int(6 * frequency_num)
    elif spa_enc_type == "theorynorm":
        in_dim = int(6 * frequency_num)
    elif spa_enc_type == "theorydiag":
        in_dim = None
    elif spa_enc_type == "naive":
        in_dim = 2
    elif spa_enc_type == "rbf":
        in_dim = num_rbf_anchor_pts
    elif spa_enc_type == "rff":
        in_dim = frequency_num
    elif spa_enc_type == "sphere":
        in_dim = int(3 * frequency_num)
    elif spa_enc_type == "spheregrid":
        in_dim = int(6 * frequency_num)
    elif spa_enc_type == "spheremixscale":
        in_dim = int(5 * frequency_num)
    elif spa_enc_type == "spheregridmixscale":
        in_dim = int(8 * frequency_num)
    elif spa_enc_type == "dft":
        in_dim = frequency_num * 4 + 4*frequency_num*frequency_num
    elif spa_enc_type == "aodha":
        in_dim = None
    elif spa_enc_type == "kdelta":
        in_dim = int(coord_dim*(k_delta+1))
    elif spa_enc_type == "none":
        in_dim = int(coord_dim)
    else:
        raise Exception("Space encoder function no support!")
    return in_dim


def get_spa_encoder(args, train_locs, spa_enc_type, spa_embed_dim, extent, coord_dim = 2,
                    frequency_num = 16, 
                    max_radius = 10000, min_radius = 1, 
                    f_act = "sigmoid", freq_init = "geometric",
                    num_rbf_anchor_pts = 100, rbf_kernal_size = 10e2,  
                    k_delta = 1,
                    use_postmat = True,
                    device = "cuda"):
    in_dim = get_spa_enc_input_dim(spa_enc_type, 
            frequency_num = frequency_num, 
            coord_dim = coord_dim, 
            num_rbf_anchor_pts = num_rbf_anchor_pts, 
            k_delta = k_delta)
    if spa_enc_type == "gridcell":
        # in_dim = int(4 * frequency_num)
        if use_postmat:
            ffn = get_ffn(args,
                input_dim = in_dim,
                output_dim = spa_embed_dim,
                f_act = f_act,
                context_str = "GridCellSpatialRelationEncoder")
        else:
            ffn = None
            assert in_dim == spa_embed_dim
        spa_enc = GridCellSpatialRelationEncoder(
            spa_embed_dim, 
            coord_dim = coord_dim, 
            frequency_num = frequency_num, 
            max_radius = max_radius,
            min_radius = min_radius,
            freq_init = freq_init,
            ffn=ffn,
            device=device)
    elif spa_enc_type == "gridcellnorm":
        # in_dim = int(4 * frequency_num)
        if use_postmat:
            ffn = get_ffn(args,
                input_dim = in_dim,
                output_dim=spa_embed_dim,
                f_act = f_act,
                context_str = "GridCellNormSpatialRelationEncoder")
        else:
            ffn = None
            assert in_dim == spa_embed_dim
        spa_enc = GridCellNormSpatialRelationEncoder(
            spa_embed_dim, 
            coord_dim = coord_dim, 
            frequency_num = frequency_num, 
            max_radius = max_radius,
            min_radius = min_radius,
            freq_init = freq_init,
            ffn=ffn,
            device=device)
    elif spa_enc_type == "hexagridcell":
        spa_enc = HexagonGridCellSpatialRelationEncoder(
            spa_embed_dim, 
            coord_dim = coord_dim, 
            frequency_num = frequency_num, 
            max_radius = max_radius,
            dropout = args.dropout, 
            f_act= f_act,
            device=device)
    elif spa_enc_type == "theory":
        # in_dim = int(6 * frequency_num)
        if use_postmat:
            ffn = get_ffn(args,
                input_dim = in_dim,
                output_dim=spa_embed_dim,
                f_act = f_act,
                context_str = "TheoryGridCellSpatialRelationEncoder")
        else:
            ffn = None
            assert in_dim == spa_embed_dim
        spa_enc = TheoryGridCellSpatialRelationEncoder(
            spa_embed_dim,
            coord_dim = coord_dim,
            frequency_num = frequency_num,
            max_radius = max_radius,
            min_radius = min_radius,
            freq_init = freq_init,
            ffn=ffn,
            device=device)
    elif spa_enc_type == "theorynorm":
        # in_dim = int(6 * frequency_num)
        if use_postmat:
            ffn = get_ffn(args,
                input_dim = in_dim,
                output_dim=spa_embed_dim,
                f_act = f_act,
                context_str = "TheoryGridCellNormSpatialRelationEncoder")
        else:
            ffn = None
            assert in_dim == spa_embed_dim
        spa_enc = TheoryGridCellNormSpatialRelationEncoder(
            spa_embed_dim,
            coord_dim = coord_dim,
            frequency_num = frequency_num,
            max_radius = max_radius,
            min_radius = min_radius,
            freq_init = freq_init,
            ffn=ffn,
            device=device)
    elif spa_enc_type == "theorydiag":
        spa_enc = TheoryDiagGridCellSpatialRelationEncoder(
            spa_embed_dim, 
            coord_dim = coord_dim, 
            frequency_num = frequency_num, 
            max_radius = max_radius, 
            min_radius = min_radius,
            dropout = args.dropout, 
            f_act= f_act, 
            freq_init = freq_init, 
            use_layn = args.use_layn, 
            use_post_mat = use_postmat,
            device=device)
    elif spa_enc_type == "naive":
        # in_dim = 2
        if use_postmat:
            ffn = get_ffn(args,
                input_dim = in_dim,
                output_dim = spa_embed_dim,
                f_act = f_act,
                context_str = "NaiveSpatialRelationEncoder")
        else:
            ffn = None
            assert in_dim == spa_embed_dim
        spa_enc = NaiveSpatialRelationEncoder(
            spa_embed_dim, 
            extent = extent, 
            coord_dim = coord_dim, 
            ffn = ffn,
            device=device)
    elif spa_enc_type == "rbf":
        # in_dim = num_rbf_anchor_pts
        if use_postmat:
            ffn = get_ffn(args,
                input_dim = in_dim,
                output_dim=spa_embed_dim,
                f_act = f_act,
                context_str = "RBFSpatialRelationEncoder")
        else:
            ffn = None
            assert in_dim == spa_embed_dim
        spa_enc = RBFSpatialRelationEncoder(
            model_type = "global", 
            train_locs = train_locs,
            spa_embed_dim = spa_embed_dim,
            coord_dim = coord_dim, 
            num_rbf_anchor_pts = num_rbf_anchor_pts,
            rbf_kernal_size = rbf_kernal_size,
            rbf_kernal_size_ratio = 0,
            max_radius = max_radius,
            ffn=ffn,
            rbf_anchor_pt_ids = args.rbf_anchor_pt_ids,
            device = device)
    elif spa_enc_type == "rff":
        # in_dim = frequency_num
        if use_postmat:
            ffn = get_ffn(args,
                input_dim = in_dim,
                output_dim=spa_embed_dim,
                f_act = f_act,
                context_str = "RFFSpatialRelationEncoder")
        else:
            ffn = None
            assert in_dim == spa_embed_dim
        spa_enc = RFFSpatialRelationEncoder(
            spa_embed_dim, 
            coord_dim = coord_dim, 
            frequency_num = frequency_num,
            rbf_kernal_size = rbf_kernal_size,
            extent = extent, 
            ffn=ffn,
            device = device)
    elif spa_enc_type == "sphere":
        # in_dim = int(3 * frequency_num)
        if use_postmat:
            ffn = get_ffn(args,
                input_dim = in_dim,
                output_dim=spa_embed_dim,
                f_act = f_act,
                context_str = "SphereSpatialRelationEncoder")
        else:
            ffn = None
            assert in_dim == spa_embed_dim
        spa_enc = SphereSpatialRelationEncoder(
            spa_embed_dim, 
            coord_dim = coord_dim, 
            frequency_num = frequency_num, 
            max_radius = max_radius,
            min_radius = min_radius,
            freq_init = freq_init,
            ffn=ffn,
            device=device)
    elif spa_enc_type == "spheregrid":
        # in_dim = int(6 * frequency_num)
        if use_postmat:
            ffn = get_ffn(args,
                input_dim = in_dim,
                output_dim=spa_embed_dim,
                f_act = f_act,
                context_str = "SphereGirdSpatialRelationEncoder")
        else:
            ffn = None
            assert in_dim == spa_embed_dim
        spa_enc = SphereGirdSpatialRelationEncoder(
            spa_embed_dim, 
            coord_dim = coord_dim, 
            frequency_num = frequency_num, 
            max_radius = max_radius,
            min_radius = min_radius,
            freq_init = freq_init,
            ffn=ffn,
            device=device)
    elif spa_enc_type == "spheremixscale":
        # in_dim = int(5 * frequency_num)
        if use_postmat:
            ffn = get_ffn(args,
                input_dim =  in_dim,
                output_dim=spa_embed_dim,
                f_act = f_act,
                context_str = "SphereMixScaleSpatialRelationEncoder")
        else:
            ffn = None
            assert in_dim == spa_embed_dim
        spa_enc = SphereMixScaleSpatialRelationEncoder(
            spa_embed_dim, 
            coord_dim = coord_dim, 
            frequency_num = frequency_num, 
            max_radius = max_radius,
            min_radius = min_radius,
            freq_init = freq_init,
            ffn=ffn,
            device=device)
    elif spa_enc_type == "spheregridmixscale":
        # in_dim = int(8 * frequency_num)
        if use_postmat:
            ffn = get_ffn(args,
                input_dim = in_dim,
                output_dim=spa_embed_dim,
                f_act = f_act,
                context_str = "SphereGridMixScaleSpatialRelationEncoder")
        else:
            ffn = None
            assert in_dim == spa_embed_dim
        spa_enc = SphereGridMixScaleSpatialRelationEncoder(
            spa_embed_dim, 
            coord_dim = coord_dim, 
            frequency_num = frequency_num, 
            max_radius = max_radius,
            min_radius = min_radius,
            freq_init = freq_init,
            ffn=ffn,
            device=device)
    elif spa_enc_type == "dft":
        # in_dim = frequency_num * 4 + 4*frequency_num*frequency_num
        if use_postmat:
            ffn = get_ffn(args,
                input_dim=int(in_dim),
                output_dim=spa_embed_dim,
                f_act = f_act,
                context_str = "DFTSpatialRelationEncoder")
        else:
            ffn = None
            assert in_dim == spa_embed_dim
        spa_enc = DFTSpatialRelationEncoder(
            spa_embed_dim, 
            coord_dim = coord_dim, 
            frequency_num = frequency_num, 
            max_radius = max_radius,
            min_radius = min_radius,
            freq_init = freq_init,
            ffn=ffn,
            device=device)
    elif spa_enc_type == "aodha":
        # extent = get_spatial_context(model_type, pointset, max_radius)
        spa_enc = AodhaSpatialRelationEncoder(
            spa_embed_dim, 
            extent = extent, 
            coord_dim = coord_dim,
            num_hidden_layers = args.num_hidden_layer,
            hidden_dim = args.hidden_dim,
            use_post_mat=use_postmat,
            f_act=f_act)
    elif spa_enc_type == "kdelta":
        # in_dim = int(coord_dim*(k_delta+1))
        if use_postmat:
            ffn = get_ffn(args,
                input_dim = in_dim,
                output_dim=spa_embed_dim,
                f_act = f_act,
                context_str = "KDeltaSpatialRelationEncoder")
        else:
            ffn = None
            assert in_dim == spa_embed_dim
        spa_enc = KDeltaSpatialRelationEncoder(
            spa_embed_dim, 
            coord_dim = coord_dim, 
            k_delta = k_delta, 
            ffn = ffn,
            device = device)
    elif spa_enc_type == "none":
        # in_dim = int(coord_dim)
        if use_postmat:
            ffn = get_ffn(args,
                input_dim = in_dim,
                output_dim=spa_embed_dim,
                f_act = f_act,
                context_str = "NoneSpatialRelationEncoder")
        else:
            ffn = None
            assert in_dim == spa_embed_dim

        spa_enc = NoneSpatialRelationEncoder(
            spa_embed_dim, 
            coord_dim = coord_dim, 
            ffn = ffn,
            device=device)
    else:
        raise Exception("Space encoder function no support!")
    return spa_enc



def get_resnet_block_by_type(resnet_block_type):
    if resnet_block_type == "basic":
        return BasicBlock
    elif resnet_block_type == "bottleneck":
        return BottleneckBlock
    else:
        raise Exception("Unknown ResNet Block type")


def compute_pgon_nuft_embed_dim(args, pgon_enc_type, freqXY):
    assert len(freqXY) == 2
    fx, fy = freqXY
    if pgon_enc_type == "nuft_ddsl":
        pgon_nuft_embed_dim = fx * (fy//2+1) * 1 * 2
    elif pgon_enc_type == "nuft_specpool":
        assert len(args.spec_pool_max_freqXY) == 2
        wx, wy = args.spec_pool_max_freqXY
        # pgon_nuft_embed_dim = fx * math.ceil(fy/2) * 1 * 2
        pgon_nuft_embed_dim = 4 * wx * wy
    elif pgon_enc_type in ["nuft_pca" , "nuftifft_pca"]:
        pgon_nuft_embed_dim = args.nuft_pca_dim
    elif pgon_enc_type == "nuftifft_mlp":
        pgon_nuft_embed_dim = fx * fy
    else:
        raise Except("pgon_enc_type unknown")
    return pgon_nuft_embed_dim


def get_resnet_polygon_encoder(args, spa_enc, pgon_embed_dim):
    resnet_block = get_resnet_block_by_type(args.resnet_block_type)
    # assert len(resnet_layers_per_block) == 3

    pgon_seq_enc = ResNet1D(block = resnet_block, 
                    num_layer_list = args.resnet_layers_per_block, 
                    in_channels = args.spa_embed_dim, 
                    out_channels = pgon_embed_dim, 
                    add_middle_pool = args.resnet_add_middle_pool,
                    final_pool = args.resnet_fl_pool_type,
                    padding_mode = args.padding_mode,
                    dropout_rate = args.dropout)

    pgon_enc = PolygonEncoder(spa_enc = spa_enc, 
                            pgon_seq_enc = pgon_seq_enc, 
                            spa_embed_dim = args.spa_embed_dim, 
                            pgon_embed_dim = pgon_embed_dim, 
                            device = args.device)
    return pgon_enc

def get_nuft_ddsl_polygon_encoder(args, pgon_enc_type, pgon_embed_dim, extent, 
            fft_sigma = 2.0, eps = 1e-6, elem_batch = 300):
    pgon_nuft_embed_dim = compute_pgon_nuft_embed_dim(args, pgon_enc_type, args.nuft_freqXY)
    ffn = get_ffn(args,
            input_dim = pgon_nuft_embed_dim,
            output_dim = pgon_embed_dim,
            f_act = args.spa_f_act,
            context_str = "NUFTPolygonEncoder")

    pgon_enc = NUFTPolygonEncoder(
                    pgon_embed_dim = pgon_embed_dim, 
                    dropout_rate = args.dropout, 
                    ffn = ffn,
                    extent = extent, 
                    eps = eps,
                    freqXY = args.nuft_freqXY, 
                    min_freqXY = args.nuft_min_freqXY, 
                    max_freqXY= args.nuft_max_freqXY, 
                    mid_freqXY = args.nuft_mid_freqXY,
                    freq_init = args.nuft_freq_init,
                    j = args.j,
                    embed_norm = args.pgon_nuft_embed_norm_type,
                    smoothing = 'gaussian', 
                    fft_sigma = fft_sigma, 
                    elem_batch = elem_batch, 
                    mode = 'density', 
                    device = args.device)
    return pgon_enc

def get_nuftifft_mlp_polygon_encoder(args, pgon_enc_type, pgon_embed_dim, extent, 
            fft_sigma = 2.0, eps = 1e-6, elem_batch = 300):
    pgon_nuft_embed_dim = compute_pgon_nuft_embed_dim(args, pgon_enc_type, args.nuft_freqXY)
    ffn = get_ffn(args,
            input_dim = pgon_nuft_embed_dim,
            output_dim = pgon_embed_dim,
            f_act = args.spa_f_act,
            context_str = "NUFTIFFTMLPPolygonEncoder")

    pgon_enc = NUFTIFFTMLPPolygonEncoder(
                    pgon_embed_dim = pgon_embed_dim, 
                    ffn = ffn,
                    extent = extent,
                    eps = eps,
                    freqXY = args.nuft_freqXY, 
                    min_freqXY = args.nuft_min_freqXY, 
                    max_freqXY= args.nuft_max_freqXY, 
                    mid_freqXY = args.nuft_mid_freqXY, 
                    freq_init = args.nuft_freq_init,
                    j = args.j,
                    embed_norm = args.pgon_nuft_embed_norm_type,
                    smoothing = 'gaussian', 
                    fft_sigma = fft_sigma, 
                    elem_batch = elem_batch, 
                    mode = 'density', 
                    device = args.device)
    return pgon_enc

def get_nuftifft_pca_polygon_encoder(args, pgon_enc_type, pgon_embed_dim, extent, 
            fft_sigma = 2.0, eps = 1e-6, elem_batch = 300):
    pgon_nuft_embed_dim = compute_pgon_nuft_embed_dim(args, pgon_enc_type, args.nuft_freqXY)
    ffn = get_ffn(args,
            input_dim = pgon_nuft_embed_dim,
            output_dim = pgon_embed_dim,
            f_act = args.spa_f_act,
            context_str = "NUFTIFFTPCAMLPPolygonEncoder")

    pgon_enc = NUFTIFFTPCAMLPPolygonEncoder(
                    pgon_embed_dim = pgon_embed_dim, 
                    ffn = ffn,
                    extent = extent, 
                    eps = eps,
                    freqXY = args.nuft_freqXY, 
                    min_freqXY = args.nuft_min_freqXY, 
                    max_freqXY= args.nuft_max_freqXY, 
                    mid_freqXY = args.nuft_mid_freqXY,
                    freq_init = args.nuft_freq_init,
                    j = args.j,
                    embed_norm = args.pgon_nuft_embed_norm_type,
                    smoothing = 'gaussian', 
                    fft_sigma = fft_sigma, 
                    elem_batch = elem_batch, 
                    mode = 'density', 
                    pca_dim = args.nuft_pca_dim,
                    pca_mat = args.pca_mat,
                    device = args.device)
    return pgon_enc

def get_nuft_specpool_polygon_encoder(args, pgon_enc_type, pgon_embed_dim, extent, 
            fft_sigma = 2.0, eps = 1e-6, elem_batch = 300):
    pgon_nuft_embed_dim = compute_pgon_nuft_embed_dim(args, pgon_enc_type, args.nuft_freqXY)
    ffn = get_ffn(args,
            input_dim = pgon_nuft_embed_dim,
            output_dim = pgon_embed_dim,
            f_act = args.spa_f_act,
            context_str = "NUFTSpecPoolPolygonEncoder")

    pgon_enc = NUFTSpecPoolPolygonEncoder(
                    pgon_embed_dim = pgon_embed_dim, 
                    dropout_rate = args.dropout, 
                    ffn = ffn,
                    extent = extent, 
                    eps = eps,
                    freqXY = args.nuft_freqXY, 
                    min_freqXY = args.nuft_min_freqXY, 
                    max_freqXY= args.nuft_max_freqXY, 
                    mid_freqXY = args.nuft_mid_freqXY,
                    freq_init = args.nuft_freq_init,
                    j = args.j,
                    embed_norm = args.pgon_nuft_embed_norm_type,
                    smoothing = 'gaussian', 
                    fft_sigma = fft_sigma, 
                    elem_batch = elem_batch, 
                    mode = 'density', 
                    spec_pool_max_freqXY = args.spec_pool_max_freqXY, 
                    spec_pool_min_freqXY_ratio  = args.spec_pool_min_freqXY_ratio,
                    device = args.device)
    return pgon_enc

def get_nuft_pca_polygon_encoder(args, pgon_enc_type, pgon_embed_dim, extent, 
            fft_sigma = 2.0, eps = 1e-6, elem_batch = 300):
    pgon_nuft_embed_dim = compute_pgon_nuft_embed_dim(args, pgon_enc_type, args.nuft_freqXY)
    ffn = get_ffn(args,
            input_dim = pgon_nuft_embed_dim,
            output_dim = pgon_embed_dim,
            f_act = args.spa_f_act,
            context_str = "NUFTPCAPolygonEncoder")

    pgon_enc = NUFTPCAPolygonEncoder(
                    pgon_embed_dim = pgon_embed_dim, 
                    dropout_rate = args.dropout, 
                    ffn = ffn,
                    extent = extent, 
                    eps = eps,
                    freqXY = args.nuft_freqXY, 
                    min_freqXY = args.nuft_min_freqXY, 
                    max_freqXY= args.nuft_max_freqXY, 
                    mid_freqXY = args.nuft_mid_freqXY,
                    freq_init = args.nuft_freq_init,
                    j = args.j,
                    embed_norm = args.pgon_nuft_embed_norm_type,
                    smoothing = 'gaussian', 
                    fft_sigma = fft_sigma, 
                    elem_batch = elem_batch, 
                    mode = 'density', 
                    nuft_pca_dim = args.nuft_pca_dim,
                    pca_mat = args.pca_mat,
                    device = args.device)
    return pgon_enc



def get_polygon_encoder(args, pgon_enc_type, spa_enc, spa_embed_dim, pgon_embed_dim, 
            extent = (-1,1,-1,1), padding_mode = "circle", 
            resnet_add_middle_pool = False, resnet_fl_pool_type = "mean", 
            resnet_block_type = "basic", resnet_layers_per_block = [2, 2, 2], 
            dropout = 0.5, 
            elem_batch = 300, fft_sigma = 2.0, eps = 1e-6, device = "cpu"):
    
    if pgon_enc_type == "resnet":
        # resnet_block = get_resnet_block_by_type(resnet_block_type)
        # # assert len(resnet_layers_per_block) == 3

        # pgon_seq_enc = ResNet1D(block = resnet_block, 
        #                 num_layer_list = resnet_layers_per_block, 
        #                 in_channels = spa_embed_dim, 
        #                 out_channels = pgon_embed_dim, 
        #                 add_middle_pool = resnet_add_middle_pool,
        #                 final_pool = resnet_fl_pool_type,
        #                 padding_mode = padding_mode,
        #                 dropout_rate = dropout)

        # pgon_enc = PolygonEncoder(spa_enc = spa_enc, 
        #                         pgon_seq_enc = pgon_seq_enc, 
        #                         spa_embed_dim = spa_embed_dim, 
        #                         pgon_embed_dim = pgon_embed_dim, 
        #                         device = device)
        pgon_enc = get_resnet_polygon_encoder(args, spa_enc, pgon_embed_dim)
    elif pgon_enc_type == "veercnn": 
        pgon_enc = VeerCNNPolygonEncoder(pgon_embed_dim = pgon_embed_dim, 
                                dropout_rate = dropout, 
                                padding_mode = padding_mode,
                                device = device)
    elif  pgon_enc_type == "nuft_ddsl":
        # pgon_nuft_embed_dim = compute_pgon_nuft_embed_dim(pgon_enc_type, args.nuft_freqXY)
        # ffn = get_ffn(args,
        #         input_dim = pgon_nuft_embed_dim,
        #         output_dim = pgon_embed_dim,
        #         f_act = args.spa_f_act,
        #         context_str = "NUFTPolygonEncoder")

        # pgon_enc = NUFTPolygonEncoder(
        #                 pgon_embed_dim = pgon_embed_dim, 
        #                 dropout_rate = dropout, 
        #                 ffn = ffn,
        #                 extent = extent, 
        #                 eps = eps,
        #                 freqXY = args.nuft_freqXY, 
        #                 j = args.j,
        #                 smoothing = 'gaussian', 
        #                 fft_sigma = fft_sigma, 
        #                 elem_batch = 300, 
        #                 mode = 'density', 
        #                 device = device)
        pgon_enc = get_nuft_ddsl_polygon_encoder(args, 
                            pgon_enc_type, 
                            pgon_embed_dim,
                            extent, 
                            fft_sigma = fft_sigma, eps = eps, 
                            elem_batch = elem_batch)
    elif pgon_enc_type == "nuftifft_ddsl":
        # assert args.model_type == "imgcat"
        pgon_enc = NUFTIFFTPolygonEncoder(
                        extent = extent, 
                        eps = eps,
                        freqXY = args.nuft_freqXY, 
                        j = args.j,
                        smoothing = 1, 
                        fft_sigma = fft_sigma, 
                        elem_batch = 300, 
                        mode = 'density', 
                        device = device)
    elif pgon_enc_type == "nuftifft_mlp":

        pgon_enc = get_nuftifft_mlp_polygon_encoder(args, 
                            pgon_enc_type, 
                            pgon_embed_dim,
                            extent, 
                            fft_sigma = fft_sigma, eps = eps, 
                            elem_batch = elem_batch)
    elif pgon_enc_type in "nuftifft_pca":
        pgon_enc = get_nuftifft_pca_polygon_encoder(args, 
                            pgon_enc_type, 
                            pgon_embed_dim,
                            extent, 
                            fft_sigma = fft_sigma, eps = eps, 
                            elem_batch = elem_batch)

    elif pgon_enc_type == "resnet__nuft_ddsl":
        pgon_enc_1 = get_resnet_polygon_encoder(args, spa_enc, pgon_embed_dim//2)
        pgon_enc_2 = get_nuft_ddsl_polygon_encoder(args, 
                            pgon_enc_type = "nuft_ddsl", 
                            pgon_embed_dim = pgon_embed_dim - pgon_embed_dim//2,
                            extent = extent, 
                            fft_sigma = fft_sigma, eps = eps, 
                            elem_batch = elem_batch)
        pgon_enc = ConcatPolygonEncoder(pgon_embed_dim, 
                            pgon_enc_1 = pgon_enc_1, 
                            pgon_enc_2 = pgon_enc_2, 
                            device = device)
    elif pgon_enc_type == "nuft_specpool":
        pgon_enc = get_nuft_specpool_polygon_encoder(args, 
                            pgon_enc_type, 
                            pgon_embed_dim,
                            extent, 
                            fft_sigma = fft_sigma, eps = eps, 
                            elem_batch = elem_batch)
    elif pgon_enc_type in "nuft_pca":
        pgon_enc = get_nuft_pca_polygon_encoder(args, 
                            pgon_enc_type, 
                            pgon_embed_dim,
                            extent, 
                            fft_sigma = fft_sigma, eps = eps, 
                            elem_batch = elem_batch)
    else:
        raise Exception("Unknow PolygonEncoder type")
    return pgon_enc


def get_polygon_decoder(args, pgon_dec_type, spa_enc, pgon_embed_dim, num_vert, 
        pgon_dec_grid_init, pgon_dec_grid_enc_type, coord_dim = 2, padding_mode = "circular", 
        extent = (-1, 1. -1, 1), device = "cpu"):
    if pgon_dec_type == "explicit_mlp":
        pgon_dec = ExplicitMLPPolygonDecoder(spa_enc = spa_enc,
                                        pgon_embed_dim = pgon_embed_dim, 
                                        num_vert = num_vert, 
                                        pgon_dec_grid_init = pgon_dec_grid_init,
                                        pgon_dec_grid_enc_type = pgon_dec_grid_enc_type,
                                        coord_dim = coord_dim, 
                                        extent = extent,
                                        device = device)
    elif pgon_dec_type == "explicit_conv":
        pgon_dec = ExplicitConvPolygonDecoder(spa_enc = spa_enc,
                                        pgon_embed_dim = pgon_embed_dim, 
                                        num_vert = num_vert, 
                                        pgon_dec_grid_init = pgon_dec_grid_init,
                                        pgon_dec_grid_enc_type = pgon_dec_grid_enc_type,
                                        coord_dim = coord_dim, 
                                        padding_mode = padding_mode,
                                        extent = extent,
                                        device = device)
    return pgon_dec

    

def get_triple_rel_dec(args, model_type, pgon_enc, 
            num_classes = 10, pgon_norm_reg_weight = 0.1, do_weight_norm = False, device = "cpu"):
    '''
    get a triple relation decoder, 
    Given subject and object polygons, predict the relation
    Args:
        model_type:
            cat:
        pgon_enc: polygon encoder
        num_classes: number of class labels

    '''
    
    if model_type == "cat":
        triple_dec = RelationConcatClassifer(pgon_enc, 
                                num_classes = num_classes, 
                                pgon_norm_reg_weight = pgon_norm_reg_weight, 
                                do_weight_norm = do_weight_norm, 
                                device = device).to(device)
    elif model_type == "imgcatmlp":
        
        triple_dec = RelationImageConcatMLPClassifer(pgon_enc, 
                                num_classes = num_classes, 
                                pgon_norm_reg_weight = pgon_norm_reg_weight, 
                                do_weight_norm = do_weight_norm,
                                num_hidden_layers = args.num_hidden_layer, 
                                hidden_dim = args.hidden_dim,
                                dropout = args.dropout,
                                f_act = args.act,
                                use_layn = args.use_layn,
                                skip_connection = args.skip_connection,
                                mean = 0, 
                                std = 1, 
                                device = device
                                ).to(device)
    elif  "imgcat" in model_type:
        if model_type == "imgcat":
            img_cla_type = "lenet5"
        else:
            _, img_cla_type = model_type.split("_")
        triple_dec = RelationImageConcatClassifer(pgon_enc, 
                                num_classes = num_classes, 
                                pgon_norm_reg_weight = pgon_norm_reg_weight, 
                                hidden_dim = args.hidden_dim,
                                mean = 0, 
                                std = 1, 
                                img_cla_type = img_cla_type,
                                device = device).to(device)
    else:
        raise Exception("Not Implement Error")
    

    return triple_dec



def get_polygon_classifer(args, pgon_enc, num_classes, 
            pgon_norm_reg_weight, do_weight_norm, hidden_dim, device = "cpu"):
    if args.model_type == "":
        pgon_classifer = PolygonClassifer(
                            pgon_enc = pgon_enc, 
                            num_classes = num_classes, 
                            pgon_norm_reg_weight = pgon_norm_reg_weight,
                            do_weight_norm = do_weight_norm,
                            device = device).to(device)
    elif args.model_type == "imgcla":
        pgon_classifer = PolygonImageClassifer(
                            pgon_enc = pgon_enc, 
                            num_classes = num_classes, 
                            pgon_norm_reg_weight = pgon_norm_reg_weight, 
                            hidden_dim = hidden_dim, 
                            mean = 0, std = 1, 
                            device = device).to(device)
    else:
        raise Exception("Undefined model_type")
    return pgon_classifer
