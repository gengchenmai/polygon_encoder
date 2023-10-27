from argparse import ArgumentParser
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from polygonembed.dataset import *
from polygonembed.SpatialRelationEncoder import *
from polygonembed.PolygonEncoder import *
from polygonembed.PolygonDecoder import *
from polygonembed.enc_dec import *

from polygonembed.utils import *
from polygonembed.model_utils import *
from polygonembed.data_util import *
from polygonembed.trainer_helper import *

# from polygonembed.train_helper import run_train, run_eval, run_eval_per_type, run_joint_train




def make_args_parser():
    parser = ArgumentParser()
    # dir
    parser.add_argument("--data_dir", type=str, default="./dbtopo/")
    parser.add_argument("--model_dir", type=str, default="./")
    parser.add_argument("--log_dir", type=str, default="./")

    #data
    parser.add_argument("--pgon_filename", type=str, default="pgon_gdf_prj.pkl")
    parser.add_argument("--geom_type_list", nargs='+', type=str, default=["origin", "norm"],
        help='''the type of geometry we need to consider:
                origin: the original polygon
                norm: the normalized polygn into (-1, 1, -1, 1)''')
    parser.add_argument("--data_split_num", type=int, default=0,
        help='''we might do multiple train/valid/test split, 
        this indicate which split we will use to train
        Note that we use 1, 0, -1 to indicate train/test/valid
            1: train
            0: test
            -1: valid ''')
    parser.add_argument("--num_worker", type=int, default=0,
        help='the number of worker for dataloader')
    parser.add_argument("--num_vert", type=int, default=1000,
        help='the number of unique vertices of one polygon')


    # model type
    parser.add_argument("--task", nargs='+', type=str, default=["grt"],
        help='''the task 
        grt: generate polygon
        cla: polygon classification
        ''')
    parser.add_argument("--model_type", type=str, default="",
        help='''the type of pololygon classifier we use, 
        "": PolygonClassifier()
        imgcla: PolygonImageClassifier()
        ''')
    

    # model
    # parser.add_argument("--embed_dim", type=int, default=64,
    #     help='Point feature embedding dim')
    parser.add_argument("--dropout", type=float, default=0.5,
        help='The dropout rate used in all fully connected layer')
    parser.add_argument("--act", type=str, default='sigmoid',
        help='the activation function for the encoder decoder')



    




    # # encoder decoder
    # parser.add_argument("--join_dec_type", type=str, default='max',
    #     help='the type of join_dec, min/max/mean/cat')

    # polygon encoder
    parser.add_argument("--pgon_enc", type=str, default="resnet",
        help='''the type of polygon encoder:
                resnet: ResNet based encoder
                veercnn: the CNN model proposed in https://arxiv.org/pdf/1806.03857.pdf
                nuft_ddsl: the NUDF DDSL model in the spectural domain
                nuftifft_ddsl: the NUDF + IFFT + LeNet5 mode in https://arxiv.org/pdf/1901.11082.pdf
                nuft_specpool: the NUFT DDSL model add a spectural pool
                nuft_pca: the NUFT DDSL followed by a PCA 
                ''')
    parser.add_argument("--nuft_pca_dim", type=int, default=64,
        help='the number of pca component we want to keep')
    parser.add_argument("--nuft_pca_white", type=str, default="nw",
        help='''nw: PCA (whiten = False)
                wb: PCA (whiten = True)''')
    parser.add_argument("--pgon_embed_dim", type=int, default=64,
        help='the embedding dimention of polygon')
    parser.add_argument("--padding_mode", type=str, default="circular",
        help='the type of padding method for Conv1D: circular / zeros / reflect / replicate')
    parser.add_argument("--resnet_add_middle_pool", type=str, default='F',
        help='whether to add MaxPool1D between the middle layers of ResNet')
    parser.add_argument("--resnet_fl_pool_type", type=str, default="mean",
        help='''the type of final pooling method: 
                mean / min /max:
                atten_[att_type]_[bn]_[nat]:
                    att_type: the type of attention
                        whole: we combine embeddings with a scalar attention coefficient
                        ele: we combine embedding with a vector attention coefficient
                    bn: the type of batch noralization type
                        no: no batch norm
                        before: batch norm before ReLU
                        after:  batch norm after ReLU
                    nat: scalar = [1,2,3], the number of attention matrix we want to go through before atten_mats2''')
    parser.add_argument("--resnet_block_type", type=str, default="basic",
        help='the type of ResNet block we will use: basic / bottleneck')
    parser.add_argument("--resnet_layers_per_block", nargs='+', type=int, default=[],
        help='the number of layers per resnet block, must be 3 layers')
    # nuft args
    parser.add_argument("--nuft_freqXY", nargs='+', type=int, default=[16, 16],
        help='the number of frequency used for each spatial dimenstion, must be 2 -> [fx, fy]')
    parser.add_argument("--nuft_max_freqXY", type=float, default=8,
        help='the max frequency we use for NUFT Fourier frequency')
    parser.add_argument("--nuft_min_freqXY", type=float, default=1,
        help='the min frequency we use for NUFT Fourier frequency')
    parser.add_argument("--nuft_mid_freqXY", type=float, default=4,
        help='the middle frequency we use for NUFT Fourier frequency')
    parser.add_argument("--nuft_freq_init", type=str, default="fft",
        help='''the frequency initilization method we use for NUFT Fourier frequency
                "geometric": geometric series
                "fft": fast fourier transformation
        ''')
    parser.add_argument("--j", type=int, default=2,
        help='the j-simplex dimention we consider')
    # parser.add_argument("--pgon_nuft_embed_norm", type=str, default='F',
    #     help='whether to normalize the polygon NUFT resulting embedding before ffn')
    parser.add_argument("--pgon_nuft_embed_norm_type", type=str, default='none',
        help='''the type of normalization for the polygon NUFT resulting embedding before ffn
        none: no norm
        l2: l2 norm
        bn: batch norm 
        ''')
    parser.add_argument("--spec_pool_max_freqXY", nargs='+', type=int, default=[16, 16],
        help='''the maximum number of spectural pooling frequency 
            used for each spectural dimenstion, must be 2 -> [fx, fy]''')
    parser.add_argument("--spec_pool_min_freqXY_ratio", type=float, default=0.3,
        help='''The minimum freq ratio = min_fx/spec_pool_max_freqXY in spectual pooling, 
            https://arxiv.org/pdf/1506.03767.pdf''')

    
    
    # parser.add_argument("--spec_pool_alpha", type=float, default=0.3,
    #     help='The alpha in spectual pooling, https://arxiv.org/pdf/1506.03767.pdf')
    # parser.add_argument("--spec_pool_beta", type=float, default=0.15,
    #     help='The beta in spectual pooling, https://arxiv.org/pdf/1506.03767.pdf')


    # space encoder
    parser.add_argument("--spa_enc", type=str, default="gridcell",
        help='the type of spatial encoder, none/naive/gridcell/hexagridcell/theory/theorydiag')
    parser.add_argument("--spa_embed_dim", type=int, default=64,
        help='Point Spatial relation embedding dim')
    parser.add_argument("--freq", type=int, default=16,
        help='The number of frequency used in the space encoder')
    parser.add_argument("--max_radius", type=float, default=10e4,
        help='The maximum spatial context radius in the space encoder')
    parser.add_argument("--min_radius", type=float, default=1.0,
        help='The minimum spatial context radius in the space encoder')
    parser.add_argument("--spa_f_act", type=str, default='sigmoid',
        help='The final activation function used by spatial relation encoder')
    parser.add_argument("--freq_init", type=str, default='geometric',
        help='The frequency list initialization method')
    # parser.add_argument("--spa_enc_use_layn", type=str, default='F',
    #     help='whether to use layer normalzation in spa_enc')
    parser.add_argument("--spa_enc_use_postmat", type=str, default='T',
        help='whether to use post matrix in spa_enc')

    # rbf
    parser.add_argument("--num_rbf_anchor_pts", type=int, default=100,
        help='The number of RBF anchor points used in the "rbf" space encoder')
    parser.add_argument("--rbf_kernal_size", type=float, default=10e2,
        help='The RBF kernal size in the "rbf" space encoder')
    parser.add_argument("--rbf_kernal_size_ratio", type=float, default=0,
        help='The RBF kernal size ratio in the relative "rbf" space encoder')

    parser.add_argument("--k_delta", type=int, default=1,
        help='The number of (deltaX, deltaY) used in the "kdelta" space encoder')
    



            

    # ffn
    parser.add_argument("--ffn_type", type=str, default="ffn",
        help='''ffn:  use MultiLayerFeedForwardNN()
                ffnf: use MultiLayerFeedForwardNNFlexible()''')
    parser.add_argument("--ffn_hidden_layers", nargs='+', type=int, default=[],
        help='a list of hidden dimention in MultiLayerFeedForwardNNFlexible in the polygon encoder')
    parser.add_argument("--num_hidden_layer", type=int, default=3,
        help='The number of hidden layer in feedforward NN in the (global) space encoder')
    parser.add_argument("--hidden_dim", type=int, default=128,
        help='The hidden dimention in feedforward NN in the (global) space encoder')
    parser.add_argument("--use_layn", type=str, default="F",
        help='use layer normalization or not in feedforward NN in the (global) space encoder')
    parser.add_argument("--skip_connection", type=str, default="F",
        help='skip connection or not in feedforward NN in the (global) space encoder')



    # polygon decoder
    parser.add_argument("--pgon_dec", type=str, default="explicit_mlp",
        help='''the type of polygon decoder to do unsupervised learning, 
            explicit_mlp: elementwise mlp for each grid point
            explicit_conv: 3 layers of convolution network
            ''')
    parser.add_argument("--pgon_dec_grid_init", type=str, default="uniform",
        help='''We generate a list of grid points for polygon decoder, the type of grid points are:
            uniform: points uniformly sampled from (-1, 1, -1, 1) 
            circle: points sampled equal-distance on a circle whose radius is randomly sampled
            kdgrid: k-d regular grid
            ''')
    parser.add_argument("--pgon_dec_grid_enc_type", type=str, default="none",
        help='''the type to encode the grid point
            none: no encoding, use the original grid point
            spa_enc: use space encoder to encode grid point before
            ''')
    
    parser.add_argument("--grt_loss_func", type=str, default="NN",
        help='''The generative loss function:
            L2: L2 distance between two corresponding points
            NN: nearest neighbor loss as Equation 1 in https://arxiv.org/pdf/1712.07262.pdf
            LOOPL2: Loop L2 distance
            ''')

    parser.add_argument("--do_weight_norm", type=str, default="T",
        help = "whether we use a weight normlized linear layer for POlygn Classification")

    parser.add_argument("--do_polygon_random_start", type=str, default="T",
        help = "whether we random pick a point on polygon to start the training for POlygn Classification")
    
    parser.add_argument("--do_data_augment", type=str, default="F",
        help = "whether do polygon data argumentation, flip, rotate, scale polygons in each batch")
    parser.add_argument("--do_online_data_augment", type=str, default="F",
        help = '''whether do online polygon data argumentation during training
            T: data augment in mini-batch
            F: data augment when loading the data
            ''')
    parser.add_argument("--data_augment_type", type=str, default="flip",
        help = '''the type of data augmentation:
            none: no data argumentation
            flp: flip
            rot: rotate
            tra: translate
            scl: scale
            noi: add white noise to polygon
            ''')
    parser.add_argument("--num_augment", type=int, default=8,
        help='The number of copies we will create we do when we do data argumentation')
    
    

    # train
    parser.add_argument("--opt", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=0.01,
        help='learning rate')
    
    parser.add_argument("--weight_decay", type=float, default=0.001,
        help='weight decay of adam optimizer')
    parser.add_argument("--task_loss_weight", type=float, default=0.8,
        help='the weight of classification loss when we do join training')
    parser.add_argument("--pgon_norm_reg_weight", type=float, default=0.1,
        help='the weight of polygon embedding norm regularizer')
    
    parser.add_argument("--grt_epoches", type=int, default=50000000,
        help='the maximum epoches for generative model converge')
    parser.add_argument("--cla_epoches", type=int, default=50000000,
        help='the maximum epoches for polygon classifier model converge')
    parser.add_argument("--max_burn_in", type=int, default=5000,
        help='the maximum iterator for relative/global model converge')
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--tol", type=float, default=0.000001)

    parser.add_argument("--balanced_train_loader", type=str, default="T",
        help = "whether we do BalancedSampler for polygon classification")
    

    parser.add_argument("--tb", type=str, default="T",
        help = "whether to log to tensorboard")

    # eval
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--val_every", type=int, default=5000)


    # load old model
    parser.add_argument("--load_model", action='store_true')
    parser.add_argument("--eval_only", action='store_true')
    parser.add_argument("--save_eval", action='store_true')

    # cuda
    parser.add_argument("--device", type=str, default="cpu")

    return parser

def check_args_correctness(args):
    # 1. check whether spa_embed_dim is corrected setted as indicated by spa_enc_type
    in_dim = get_spa_enc_input_dim(
            spa_enc_type = args.spa_enc, 
            frequency_num = args.freq, 
            coord_dim = 2, 
            num_rbf_anchor_pts = args.num_rbf_anchor_pts, 
            k_delta = args.k_delta)
    if in_dim is not None and args.spa_enc_use_postmat != "T":
        # if we have in_dim and we do not want ffn
        # we need to check
        args.spa_embed_dim = in_dim

    # data argumentation
    if args.data_augment_type == "none":
        args.do_data_augment = "F"
        args.do_online_data_augment = "F"
    return args


def bool_arg_handler(arg):
    return True if arg == "T" else False

def update_args(args):
    select_args = ["use_layn", "skip_connection", "spa_enc_use_postmat", "balanced_train_loader", "do_weight_norm", 
                "do_polygon_random_start", "do_data_augment", "do_online_data_augment", "resnet_add_middle_pool", "tb"]
    for arg in select_args:
        args.__dict__[arg] = bool_arg_handler(getattr(args, arg))
    # args.use_layn = True if args.use_layn == "T" else False

    # args.skip_connection = True if args.skip_connection == "T" else False

    # args.spa_enc_use_postmat = True if args.spa_enc_use_postmat == "T" else False

    # args.balanced_train_loader = True if args.balanced_train_loader == "T" else False

    # args.do_weight_norm = True if args.do_weight_norm == "T" else False
    
    return args

def make_args_spa_enc(args):
    if args.spa_enc == "none":
        args_spa_enc = "-{spa_enc:s}-{spa_embed_dim:d}".format(
            spa_enc=args.spa_enc,
            spa_embed_dim=args.spa_embed_dim)
    elif args.spa_enc == "kdelta":
        args_spa_enc = "-{spa_enc:s}-{spa_embed_dim:d}-{k_delta:d}".format(
            spa_enc=args.spa_enc,
            spa_embed_dim=args.spa_embed_dim,
            k_delta = args.k_delta)
    else:
        args_spa_enc = "-{spa_enc:s}-{spa_embed_dim:d}-{freq:d}-{max_radius:.1f}-{min_radius:.1f}-{spa_f_act:s}-{freq_init:s}-{spa_enc_use_postmat:s}".format(
                spa_enc=args.spa_enc,
                spa_embed_dim=args.spa_embed_dim,
                freq=args.freq,
                max_radius=args.max_radius,
                min_radius=args.min_radius,
                spa_f_act=args.spa_f_act,
                freq_init=args.freq_init,
                spa_enc_use_postmat=args.spa_enc_use_postmat)

        if args.spa_enc == "rbf":
            args_spa_enc += "-{num_rbf_anchor_pts:d}-{rbf_kernal_size:.1f}-{rbf_kernal_size_ratio:.2f}".format(
                num_rbf_anchor_pts = args.num_rbf_anchor_pts,
                rbf_kernal_size=args.rbf_kernal_size,
                rbf_kernal_size_ratio=args.rbf_kernal_size_ratio
                )
    return args_spa_enc

def make_args_pgon_enc(args):
    if args.pgon_enc == "resnet":
        args_pgon_enc = "-{pgon_enc:s}-{pgon_embed_dim:d}-{padding_mode:s}-{resnet_add_middle_pool:s}-{resnet_fl_pool_type:s}-{resnet_block_type:s}-{resnet_layers_per_block:s}".format(
                pgon_enc = args.pgon_enc, 
                pgon_embed_dim = args.pgon_embed_dim, 
                padding_mode = args.padding_mode,
                resnet_add_middle_pool = args.resnet_add_middle_pool,
                resnet_fl_pool_type = args.resnet_fl_pool_type,
                resnet_block_type = args.resnet_block_type, 
                resnet_layers_per_block = "_".join([str(i) for i in args.resnet_layers_per_block]) )
    
    elif args.pgon_enc == "veercnn":
        args_pgon_enc = "-{pgon_enc:s}-{pgon_embed_dim:d}-{padding_mode:s}".format(
                pgon_enc = args.pgon_enc, 
                pgon_embed_dim = args.pgon_embed_dim,
                padding_mode = args.padding_mode )
    elif "nuft" in args.pgon_enc:
        args_pgon_enc = "-{pgon_enc:s}-{pgon_embed_dim:d}-{nuft_freqXY:s}-{j:d}-{pgon_nuft_embed_norm_type:s}".format(
                pgon_enc = args.pgon_enc, 
                pgon_embed_dim = args.pgon_embed_dim,
                nuft_freqXY = "_".join([str(i) for i in args.nuft_freqXY]),
                j = args.j,
                pgon_nuft_embed_norm_type = args.pgon_nuft_embed_norm_type )
        if args.pgon_enc == "nuft_specpool":
            args_pgon_enc += "-{spec_pool_max_freqXY:s}-{spec_pool_min_freqXY_ratio:.3f}".format(
                spec_pool_max_freqXY = "_".join([str(i) for i in args.spec_pool_max_freqXY]) , 
                spec_pool_min_freqXY_ratio = args.spec_pool_min_freqXY_ratio
                )
        if args.pgon_enc in ["nuft_pca", "nuftifft_pca"]:
            args_pgon_enc += "-{nuft_pca_dim:d}-{nuft_pca_white:s}".format(
                nuft_pca_dim = args.nuft_pca_dim,
                nuft_pca_white = args.nuft_pca_white
                )
        if args.nuft_freq_init != "fft":
            if args.nuft_freq_init == "geometric":
                args_pgon_enc += "-{nuft_freq_init:s}-{nuft_min_freqXY:s}-{nuft_max_freqXY:s}".format(
                    nuft_freq_init = args.nuft_freq_init,
                    nuft_min_freqXY = format_int_and_float(args.nuft_min_freqXY),
                    nuft_max_freqXY = format_int_and_float(args.nuft_max_freqXY)
                    )
            elif args.nuft_freq_init in ["arith_geometric", "geometric_arith"]:
                args_pgon_enc += "-{nuft_freq_init:s}-{nuft_min_freqXY:s}-{nuft_mid_freqXY:s}-{nuft_max_freqXY:s}".format(
                    nuft_freq_init = args.nuft_freq_init,
                    nuft_min_freqXY = format_int_and_float(args.nuft_min_freqXY),
                    nuft_mid_freqXY = format_int_and_float(args.nuft_mid_freqXY),
                    nuft_max_freqXY = format_int_and_float(args.nuft_max_freqXY)
                    )
    return args_pgon_enc

def format_int_and_float(x):
    if int(x) == x:
        # x is integer
        return str(int(x))
    else:
        return "{:.1f}".format(x)

def make_args_ffn(args):
    if args.ffn_type == "ffn":
        args_ffn = "-{num_hidden_layer:d}-{hidden_dim:d}-{use_layn:s}-{skip_connection:s}".format(
                num_hidden_layer = args.num_hidden_layer, 
                hidden_dim = args.hidden_dim, 
                use_layn = args.use_layn, 
                skip_connection = args.skip_connection)
    elif args.ffn_type == "ffnf":
        args_ffn = "-{ffn_hidden_layers:s}-{use_layn:s}-{skip_connection:s}".format(
                ffn_hidden_layers = "_".join([str(i) for i in args.ffn_hidden_layers]),
                use_layn = args.use_layn, 
                skip_connection = args.skip_connection)
    return args_ffn

def make_args_combine(args):
    args_data = "{data:s}-{geom_type_list:s}-{data_split_num:d}".format(
            data=args.data_dir.strip().split("/")[-2],
            # pgon_filename = args.pgon_filename.split(".")[0], 
            geom_type_list = "_".join(args.geom_type_list), 
            data_split_num = args.data_split_num
        )

    args_pgon_enc = make_args_pgon_enc(args)
    
    args_spa_enc = make_args_spa_enc(args)
        

    args_ffn = make_args_ffn(args)

    args_pgon_dec = "-{pgon_dec:s}-{pgon_dec_grid_init:s}-{pgon_dec_grid_enc_type:s}-{grt_loss_func:s}".format(
            pgon_dec = args.pgon_dec,
            pgon_dec_grid_init = args.pgon_dec_grid_init,
            pgon_dec_grid_enc_type = args.pgon_dec_grid_enc_type,
            grt_loss_func = args.grt_loss_func
            )

    args_train = "-{act:s}-{dropout:.1f}-{batch_size:d}-{lr:.6f}-{opt:s}-{weight_decay:.2f}-{task_loss_weight:.2f}-{pgon_norm_reg_weight:.2f}-{balanced_train_loader:s}".format(
            act = args.act,
            dropout=args.dropout,
            opt = args.opt,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay = args.weight_decay,
            task_loss_weight = args.task_loss_weight,
            pgon_norm_reg_weight = args.pgon_norm_reg_weight,
            balanced_train_loader = args.balanced_train_loader
            )

    args_data_augment = "-{do_polygon_random_start:s}-{do_data_augment:s}-{do_online_data_augment:s}-{data_augment_type:s}-{num_augment:d}".format(
            do_polygon_random_start = args.do_polygon_random_start,
            do_data_augment = args.do_data_augment,
            do_online_data_augment = args.do_online_data_augment,
            data_augment_type = args.data_augment_type,
            num_augment = args.num_augment
            ) 

    args_combine = "/{args_data:s}-{task:s}-{model_type:s}-{args_pgon_enc:s}-{args_spa_enc:s}-{args_ffn:s}-{args_pgon_dec:s}-{args_data_augment:s}-{args_train:s}".format(
            args_data = args_data,
            task = "_".join(args.task),
            model_type=args.model_type,

            args_pgon_enc = args_pgon_enc,

            args_spa_enc=args_spa_enc,

            args_ffn=args_ffn,
            args_pgon_dec = args_pgon_dec,
            args_data_augment = args_data_augment,
            args_train = args_train
            
            )
    return args_combine

class Trainer():
    """
    Trainer
    """
    def __init__(self, args, pgon_gdf, console = True):
        

        args = check_args_correctness(args)
 
        self.args_combine = make_args_combine(args) #+ ".L2"

        self.log_file = args.log_dir + self.args_combine + ".log"
        self.model_file = args.model_dir + self.args_combine + ".pth"
        # tensorboard log directory
        # self.tb_log_dir = args.model_dir + self.args_combine
        self.tb_log_dir = args.model_dir + "/tb"

        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

        self.logger = setup_logging(self.log_file, console = console, filemode='a')

        self.get_pgon_flag(args)

        

        self.pgon_gdf = pgon_gdf
        self.geom_type_list = args.geom_type_list
        self.extent = self.get_extent(pgon_gdf, self.geom_type_list)


        args.num_vert = self.get_num_vert(pgon_gdf, self.geom_type_list, 
                                        pgon_flag = self.pgon_flag)
        args = update_args(args)
        self.load_pca_matrix(args)

        self.args = args
        
        if "grt" in args.task:
            self.pgon_dataset, self.pgon_dataloader = self.load_polygon_dataset_dataloader(
                                                                    pgon_gdf = pgon_gdf,
                                                                    geom_type_list = self.geom_type_list, 
                                                                    num_worker = args.num_worker, 
                                                                    batch_size = args.batch_size, 
                                                                    shuffle = True, 
                                                                    id_col = "ID", 
                                                                    do_data_augment = True if args.do_data_augment and not args.do_online_data_augment else False,
                                                                    num_augment = args.num_augment,
                                                                    device = args.device,
                                                                    pgon_flag = self.pgon_flag)
        else:
            self.pgon_dataset, self.pgon_dataloader = (None, None)
        if "cla" in args.task:
            self.pgon_cla_dataset, self.pgon_cla_dataloader, self.split2num, self.data_split_col, self.train_sampler, self.num_classes = self.load_polygon_cla_dataset_dataloader(
                                                                    pgon_gdf = pgon_gdf,
                                                                    geom_type_list = self.geom_type_list,
                                                                    num_worker = args.num_worker, 
                                                                    batch_size = args.batch_size, 
                                                                    balanced_train_loader = args.balanced_train_loader,
                                                                    id_col = "ID",
                                                                    class_col = "TYPEID",
                                                                    data_split_num = args.data_split_num,
                                                                    do_data_augment = True if args.do_data_augment and not args.do_online_data_augment else False,
                                                                    num_augment = args.num_augment,
                                                                    device = args.device,
                                                                    pgon_flag = self.pgon_flag
                                                                    )
        else:
            self.pgon_cla_dataset, self.pgon_cla_dataloader  = (None, None) 
            self.split2num, self.data_split_col, self.train_sampler, self.num_classes = (None, None, None, None)

        

        self.get_model(args)
        


        if args.opt == "sgd":
            self.optimizer = optim.SGD(filter(lambda p : p.requires_grad, self.model.parameters()), lr=args.lr, momentum=0)
        elif args.opt == "adam":
            self.optimizer = optim.Adam(filter(lambda p : p.requires_grad, self.model.parameters()), lr=args.lr, weight_decay = args.weight_decay)

        print("create model from {}".format(self.args_combine + ".pth"))
        self.logger.info("Save file at {}".format(self.args_combine + ".pth"))

        if args.tb:
            self.tb_writer = SummaryWriter(self.tb_log_dir)
        else:
            self.tb_writer = None
        self.global_batch_idx = 0

    def get_pgon_flag(self, args):
        if "complex" in args.pgon_filename:
            self.pgon_flag = "complex"
        else:
            self.pgon_flag = "simple"

    def get_encoder(self, args, geom_type):
        max_radius, min_radius = scale_max_and_min_radius(geom_type, 
                                extent = self.extent[geom_type], 
                                max_radius = args.max_radius,
                                min_radius = args.min_radius)
        spa_enc = get_spa_encoder(
                                args,
                                train_locs = [],
                                spa_enc_type=args.spa_enc, 
                                spa_embed_dim=args.spa_embed_dim, 
                                extent = self.extent[geom_type],
                                coord_dim = 2, 
                                frequency_num = args.freq, 
                                max_radius = max_radius,
                                min_radius = min_radius,
                                f_act = args.spa_f_act,
                                freq_init = args.freq_init,
                                num_rbf_anchor_pts = args.num_rbf_anchor_pts,
                                rbf_kernal_size = args.rbf_kernal_size,
                                k_delta = args.k_delta,
                                use_postmat=args.spa_enc_use_postmat,
                                device = args.device)

        pgon_enc = get_polygon_encoder(
                                args, 
                                pgon_enc_type = args.pgon_enc, 
                                spa_enc = spa_enc, 
                                spa_embed_dim = args.spa_embed_dim, 
                                pgon_embed_dim = args.pgon_embed_dim, 
                                extent = self.extent[geom_type],
                                padding_mode = args.padding_mode,
                                resnet_add_middle_pool = args.resnet_add_middle_pool,
                                resnet_fl_pool_type = args.resnet_fl_pool_type,
                                resnet_block_type = args.resnet_block_type, 
                                resnet_layers_per_block = args.resnet_layers_per_block, 
                                dropout = args.dropout,
                                device = args.device)
        return spa_enc, pgon_enc

    def get_model(self, args):
        '''
        Make neural network architecture, the final output is model which content every component
        '''
        self.model = nn.Module()
        if "norm" in self.geom_type_list:
            self.norm_spa_enc, self.norm_pgon_enc = self.get_encoder(args, geom_type = "norm")
        else:
            self.norm_spa_enc, self.norm_pgon_enc = (None, None)
        if "origin" in self.geom_type_list:
            self.origin_spa_enc, self.origin_pgon_enc = self.get_encoder(args, geom_type = "origin")
        else:
            self.origin_spa_enc, self.origin_pgon_enc = (None, None)

        if "norm" in self.geom_type_list and "grt" in args.task:
            assert args.pgon_enc != "nuftifft_ddsl"
            self.norm_pgon_dec = get_polygon_decoder(
                                args, 
                                pgon_dec_type = args.pgon_dec, 
                                spa_enc = self.norm_spa_enc,
                                pgon_embed_dim = args.pgon_embed_dim, 
                                num_vert = args.num_vert, 
                                pgon_dec_grid_init = args.pgon_dec_grid_init,
                                pgon_dec_grid_enc_type = args.pgon_dec_grid_enc_type,
                                coord_dim = 2, 
                                padding_mode = "circular",
                                extent = self.extent["norm"],
                                device = args.device)


            self.model.pgon_enc_dec = PolygonEncoderDecoder(
                                pgon_enc = self.norm_pgon_enc, 
                                pgon_dec = self.norm_pgon_dec, 
                                loss_func = args.grt_loss_func,
                                device = args.device).to(args.device)
        else:
            self.norm_pgon_dec, self.model.pgon_enc_dec = (None, None)
        if "norm" in self.geom_type_list and "cla" in args.task:
            self.model.pgon_classifer = get_polygon_classifer(args,
                                    pgon_enc = self.norm_pgon_enc, 
                                    num_classes = self.num_classes, 
                                    pgon_norm_reg_weight = args.pgon_norm_reg_weight,
                                    do_weight_norm = args.do_weight_norm,
                                    hidden_dim = args.hidden_dim,
                                    device = args.device)
            # if args.model_type == "":
            #     self.model.pgon_classifer = PolygonClassifer(
            #                         pgon_enc = self.norm_pgon_enc, 
            #                         num_classes = self.num_classes, 
            #                         pgon_norm_reg_weight = args.pgon_norm_reg_weight,
            #                         do_weight_norm = args.do_weight_norm,
            #                         device = args.device).to(args.device)
            # elif args.model_type == "imgcla":
            #     self.model.pgon_classifer = PolygonImageClassifer(
            #                         pgon_enc = self.norm_pgon_enc, 
            #                         num_classes = self.num_classes, 
            #                         pgon_norm_reg_weight = args.pgon_norm_reg_weight, 
            #                         hidden_dim = args.hidden_dim, 
            #                         mean = 0, std = 1, 
            #                         device = args.device).to(args.device)
        else:
            self.model.pgon_classifer = None

        return

    def run_train(self, save_eval = False):
        # assert "norm" in self.geom_type_list
        self.global_batch_idx, eval_dict = train_model(
                    args = self.args,
                    model = self.model, 
                    pgon_dataloader = self.pgon_dataloader['norm'] if "grt" in self.args.task else None, 
                    pgon_cla_dataloader = self.pgon_cla_dataloader if "cla" in self.args.task else None,  
                    task = self.args.task,
                    task_loss_weight = self.args.task_loss_weight,
                    optimizer = self.optimizer, 
                    tb_writer = self.tb_writer, 
                    logger = self.logger,
                    model_file = self.model_file,
                    grt_epoches = self.args.grt_epoches, 
                    cla_epoches = self.args.cla_epoches,
                    log_every = self.args.log_every,
                    val_every = self.args.val_every,
                    do_polygon_random_start = self.args.do_polygon_random_start,
                    do_data_augment = self.args.do_online_data_augment,
                    global_batch_idx = self.global_batch_idx,
                    save_eval = save_eval,
                    pgon_flag = self.pgon_flag)

    def run_eval(self, save_eval = False):
        eval_dict = eval_pgon_cla_model(
                    args = self.args,
                    model = self.model, 
                    optimizer = self.optimizer, 
                    model_file = self.model_file, 
                    pgon_cla_dataloader = self.pgon_cla_dataloader,
                    logger = self.logger,
                    epoch = 0, 
                    grt_epoches = 0, 
                    global_batch_idx = self.global_batch_idx, 
                    save_eval = save_eval,
                    pgon_flag = self.pgon_flag)
        return eval_dict


    def get_num_vert(self, pgon_gdf, geom_type_list, pgon_flag = 'simple'):
        assert pgon_gdf is not None
        if pgon_flag == 'simple':
            if "norm" in geom_type_list:
                num_vert = len(pgon_gdf["geometry_norm"].iloc[0].exterior.coords) - 1
            elif "origin" in geom_type_list:
                num_vert = len(pgon_gdf["geometry"].iloc[0].exterior.coords) - 1
        elif pgon_flag == 'complex':
            if "norm" in geom_type_list:
                num_vert = count_vert(pgon_gdf["geometry_norm"].iloc[0])
            elif "origin" in geom_type_list:
                    num_vert = count_vert(pgon_gdf["geometry"].iloc[0])
        
        return num_vert
    
    def load_polygon_dataset_dataloader(self, pgon_gdf, geom_type_list, num_worker, batch_size, shuffle, id_col = "ID", 
                        do_data_augment = False, num_augment = 8, device = "cpu",
                        pgon_flag = "simple"):
        '''
        load polygon generation dataset
        '''
        # self.pgon_gdf = load_polygon_dataframe(data_dir, pgon_filename)
        pgon_dataset  = dict()
        pgon_dataloader = dict()
        for geom_type in geom_type_list:
            if pgon_flag == "simple":
                pgon_dataset[geom_type] = PolygonDataset(pgon_gdf, 
                                                        id_col = id_col, 
                                                        geom_type = geom_type, 
                                                        class_col = None, 
                                                        do_data_augment = do_data_augment,
                                                        num_augment  = num_augment,
                                                        device = device)
            elif pgon_flag == "complex":
                pgon_dataset[geom_type] = PolygonComplexDataset(pgon_gdf = pgon_gdf, 
                                                        id_col = id_col, 
                                                        geom_type = geom_type, 
                                                        class_col = None, 
                                                        do_data_augment = do_data_augment, 
                                                        num_augment = num_augment, 
                                                        device = device)
            else:
                raise Exception("Unknow pgon_flag")
            pgon_dataloader[geom_type] = torch.utils.data.DataLoader(pgon_dataset[geom_type], 
                                                        num_workers = num_worker, 
                                                        batch_size = batch_size,
                                                        shuffle = shuffle)
        return pgon_dataset, pgon_dataloader



    def load_polygon_cla_dataset_dataloader(self, pgon_gdf, geom_type_list, 
                    num_worker, batch_size, balanced_train_loader,
                    id_col = "ID", class_col = "TYPEID", data_split_num = 0, 
                    do_data_augment = False, num_augment = 8, device = "cpu",
                    pgon_flag = "simple"):
        '''
        load polygon classification dataset including training, validation, testing
        '''
        pgon_cla_dataset  = dict()
        pgon_cla_dataloader = dict()
        # the polygon classification dataset should be done on normlized polygons
        assert "norm" in geom_type_list
        data_split_col = "SPLIT_{:d}".format(data_split_num)
        split2num = {"TRAIN": 1, "TEST": 0, "VALID": -1}
        split_nums = np.unique(np.array(pgon_gdf[data_split_col]))
        assert 1 in split_nums and 0 in split_nums
        dup_test = False
        if -1 not in split_nums:
            # we will make valid and test the same
            dup_test = True

        un_class = np.unique(np.array(pgon_gdf[class_col]))
        num_classes = len(un_class)
        max_num_exs_per_class = math.ceil(batch_size/num_classes)


        for split in ["TRAIN", "TEST", "VALID"]:
            # make dataset

            if split == "VALID" and dup_test:
                pgon_cla_dataset[split] = pgon_cla_dataset["TEST"]
            else:
                pgon_split_gdf = pgon_gdf[ pgon_gdf[data_split_col] == split2num[split] ]

                if split == "TRAIN": 
                    if pgon_flag == "simple":
                        pgon_cla_dataset[split] = PolygonDataset(pgon_gdf = pgon_split_gdf, 
                                                                id_col = id_col, 
                                                                geom_type = "norm", 
                                                                class_col = class_col,
                                                                do_data_augment = do_data_augment, 
                                                                num_augment = num_augment,
                                                                device = device)
                    elif pgon_flag == "complex":
                        pgon_cla_dataset[split] = PolygonComplexDataset(pgon_gdf = pgon_split_gdf, 
                                                                id_col = id_col, 
                                                                geom_type = "norm", 
                                                                class_col = class_col, 
                                                                do_data_augment = do_data_augment, 
                                                                num_augment = num_augment, 
                                                                device = device)
                    else:
                        raise Exception("Unknow pgon_flag")
                else:
                    if pgon_flag == "simple":
                        pgon_cla_dataset[split] = PolygonDataset(pgon_gdf = pgon_split_gdf, 
                                                                id_col = id_col, 
                                                                geom_type = "norm", 
                                                                class_col = class_col,
                                                                do_data_augment = False,
                                                                device = device)
                    elif pgon_flag == "complex":
                        pgon_cla_dataset[split] = PolygonComplexDataset(pgon_gdf = pgon_split_gdf, 
                                                                id_col = id_col, 
                                                                geom_type = "norm", 
                                                                class_col = class_col, 
                                                                do_data_augment = False, 
                                                                num_augment = 0, 
                                                                device = device)
                    else:
                        raise Exception("Unknow pgon_flag")
                
            # make dataloader
            if split == "TRAIN":
                if balanced_train_loader:
                    train_sampler = BalancedSampler(classes = pgon_cla_dataset["TRAIN"].class_list.cpu().numpy(), 
                                                num_per_class = max_num_exs_per_class, 
                                                use_replace=False, 
                                                multi_label=False)
                    pgon_cla_dataloader[split] = torch.utils.data.DataLoader(pgon_cla_dataset[split], 
                                                            num_workers = num_worker, 
                                                            batch_size = batch_size,
                                                            sampler=train_sampler, 
                                                            shuffle = False)
                else:
                    train_sampler = None
                    pgon_cla_dataloader[split] = torch.utils.data.DataLoader(pgon_cla_dataset[split], 
                                                            num_workers = num_worker, 
                                                            batch_size = batch_size, 
                                                            shuffle = True)
            elif split == "VALID" and dup_test:
                pgon_cla_dataloader[split] = pgon_cla_dataloader['TEST']
            else:
                pgon_cla_dataloader[split] = torch.utils.data.DataLoader(pgon_cla_dataset[split], 
                                                            num_workers = num_worker, 
                                                            batch_size = batch_size,
                                                            shuffle = False)


        return pgon_cla_dataset, pgon_cla_dataloader, split2num, data_split_col, train_sampler, num_classes

    

    def get_extent(self, pgon_gdf, geom_type_list):
        extent = dict()
        for geom_type in geom_type_list: 
            extent[geom_type] = get_extent_by_geom_type(pgon_gdf, geom_type = geom_type)
        return extent

    def load_model(self):
        self.model, self.optimizer, self.args = load_model(self.model, self.optimizer, self.model_file)


    def load_pca_matrix(self, args):
        self.pca_mat_file = None

        pgon_filename = args.pgon_filename.split("_test")[0]

        if args.pgon_enc == "nuft_pca":
            fx, fy = args.nuft_freqXY
            F_dim = fx * (fy//2+1) * 1 * 2
            if args.nuft_freq_init == "fft":
                self.pca_mat_file = f'''{os.path.join(args.data_dir, pgon_filename).replace(".pkl", "")}-pca_{args.nuft_freqXY[0]}_{args.nuft_pca_white}.npy'''
            elif args.nuft_freq_init == "geometric":
                self.pca_mat_file = '''{file_prefix:s}-pca_{freqXY:d}_{whiten_flag:s}-{nuft_freq_init:s}-{nuft_min_freqXY:s}-{nuft_max_freqXY:s}.npy'''.format(
                    file_prefix = os.path.join(args.data_dir, pgon_filename).replace(".pkl", ""),
                    freqXY = args.nuft_freqXY[0],
                    whiten_flag = args.nuft_pca_white,
                    nuft_freq_init = args.nuft_freq_init,
                    nuft_min_freqXY = format_int_and_float(args.nuft_min_freqXY),
                    nuft_max_freqXY = format_int_and_float(args.nuft_max_freqXY)
                )


            
        elif args.pgon_enc == "nuftifft_pca":
            fx, fy = args.nuft_freqXY
            F_dim = fx * fy
            if args.nuft_freq_init == "fft":
                self.pca_mat_file = f'''{os.path.join(args.data_dir, pgon_filename).replace(".pkl", "")}-nuftifft_pca_{args.nuft_freqXY[0]}_{args.nuft_pca_white}.npy'''
            

        if self.pca_mat_file is not None:
            # pca_mat: shape (n_features, n_components), 
            #   n_components == n_features is the maximum possible nuft_pca_dim we can use
            self.pca_mat = np.load(self.pca_mat_file, allow_pickle = True)
            args.pca_mat = self.pca_mat

            # make sure the PCA matrix has the same n_features as the NUFT results
            n_features, n_components = args.pca_mat.shape
            
            assert n_features == n_components == F_dim
