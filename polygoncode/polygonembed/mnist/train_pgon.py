import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d


import functools
from functools import partial
from dataclasses import dataclass
from collections import OrderedDict


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
from polygonembed.PolygonEncoder import *
from polygonembed.utils import *
from polygonembed.data_util import *
from polygonembed.dataset import *
from polygonembed.trainer import *
from polygonembed.trainer_helper import *

from polygonembed.ddsl_utils import *
from polygonembed.ddsl import *



parser = make_args_parser()
args = parser.parse_args()

pgon_gdf = load_dataframe(args.data_dir, args.pgon_filename)

trainer = Trainer(args, pgon_gdf, console = True)

if args.load_model:
	print("load_model...")
	trainer.load_model()

if not args.eval_only:
	trainer.run_train(save_eval = args.save_eval)
else:
	trainer.run_eval(save_eval = args.save_eval)