

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



parser = make_args_parser()
args = parser.parse_args()


pgon_gdf = load_dataframe(args.data_dir, args.pgon_filename)


trainer = Trainer(args, pgon_gdf, console = True)


trainer.run_polygon_generative_train()