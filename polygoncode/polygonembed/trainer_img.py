from argparse import ArgumentParser
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from polygonembed.dataset import *
from polygonembed.resnet2d import *
from polygonembed.model_utils import *
from polygonembed.data_util import *
from polygonembed.trainer_helper import *




def make_args_parser():
    parser = ArgumentParser()
    # dir
    parser.add_argument("--data_dir", type=str, default="./animal/")
    parser.add_argument("--model_dir", type=str, default="./model_dir/animal_img/")
    parser.add_argument("--log_dir", type=str, default="./model_dir/animal_img/")

    #data
    parser.add_argument("--img_filename", type=str, default="animal_img_mats.pkl")
    
    parser.add_argument("--data_split_num", type=int, default=0,
        help='''we might do multiple train/valid/test split, 
        this indicate which split we will use to train
        Note that we use 1, 0, -1 to indicate train/test/valid
            1: train
            0: test
            -1: valid ''')
    parser.add_argument("--num_worker", type=int, default=0,
        help='the number of worker for dataloader')
    


    # model type
    parser.add_argument("--model_type", type=str, default="",
        help='''the type of image classification model we use, 
        	resnet18 / resnet34 / resnet50 / resnet101 / resnet152
        ''')
    

    # model
    # parser.add_argument("--embed_dim", type=int, default=64,
    #     help='Point feature embedding dim')
    # parser.add_argument("--dropout", type=float, default=0.5,
    #     help='The dropout rate used in all fully connected layer')
    # parser.add_argument("--act", type=str, default='sigmoid',
    #     help='the activation function for the encoder decoder')



    # # # encoder decoder
    # # parser.add_argument("--join_dec_type", type=str, default='max',
    # #     help='the type of join_dec, min/max/mean/cat')

    # # polygon encoder
    # parser.add_argument("--pgon_enc", type=str, default="resnet",
    #     help='''the type of polygon encoder:
    #             resnet: ResNet based encoder
    #             veercnn: the CNN model proposed in https://arxiv.org/pdf/1806.03857.pdf''')
    # parser.add_argument("--pgon_embed_dim", type=int, default=64,
    #     help='the embedding dimention of polygon')
    # parser.add_argument("--padding_mode", type=str, default="circular",
    #     help='the type of padding method for Conv1D: circular / zeros / reflect / replicate')
    # parser.add_argument("--resnet_add_middle_pool", type=str, default='F',
    #     help='whether to add MaxPool1D between the middle layers of ResNet')
    # parser.add_argument("--resnet_fl_pool_type", type=str, default="mean",
    #     help='the type of final pooling method: mean / min /max')
    # parser.add_argument("--resnet_block_type", type=str, default="basic",
    #     help='the type of ResNet block we will use: basic / bottleneck')
    # parser.add_argument("--resnet_layers_per_block", nargs='+', type=int, default=[],
    #     help='the number of layers per resnet block, must be 3 layers')


    
    parser.add_argument("--do_data_augment", type=str, default="F",
        help = "whether do polygon data argumentation, flip, rotate, scale polygons in each batch")
    
    
    

    # train
    parser.add_argument("--opt", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=0.01,
        help='learning rate')
    
    parser.add_argument("--weight_decay", type=float, default=0.001,
        help='weight decay of adam optimizer')
    # parser.add_argument("--task_loss_weight", type=float, default=0.8,
    #     help='the weight of classification loss when we do join training')
    # parser.add_argument("--pgon_norm_reg_weight", type=float, default=0.1,
    #     help='the weight of polygon embedding norm regularizer')
    
    # parser.add_argument("--grt_epoches", type=int, default=50000000,
    #     help='the maximum epoches for generative model converge')
    parser.add_argument("--cla_epoches", type=int, default=50000000,
        help='the maximum epoches for polygon classifier model converge')
    # parser.add_argument("--max_burn_in", type=int, default=5000,
    #     help='the maximum iterator for relative/global model converge')
    parser.add_argument("--batch_size", type=int, default=512)
    # parser.add_argument("--tol", type=float, default=0.000001)

    parser.add_argument("--balanced_train_loader", type=str, default="T",
        help = "whether we do BalancedSampler for polygon classification")
    


    # eval
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--val_every", type=int, default=5000)


    # load old model
    parser.add_argument("--load_model", action='store_true')

    # cuda
    parser.add_argument("--device", type=str, default="cpu")

    return parser


def bool_arg_handler(arg):
    return True if arg == "T" else False

def update_args(args):
    select_args = ["balanced_train_loader", "do_data_augment"]
    for arg in select_args:
        args.__dict__[arg] = bool_arg_handler(getattr(args, arg))

    return args


def make_args_combine(args):
    args_data = "{data:s}-{data_split_num:d}".format(
            data=args.data_dir.strip().split("/")[-2],
            data_split_num = args.data_split_num
        )

    args_train = "-{batch_size:d}-{lr:.6f}-{opt:s}-{weight_decay:.2f}-{balanced_train_loader:s}".format(
            # act = args.act,
            # dropout=args.dropout,
            batch_size=args.batch_size,
            lr=args.lr,
            opt = args.opt,
            weight_decay = args.weight_decay,
            # task_loss_weight = args.task_loss_weight,
            # pgon_norm_reg_weight = args.pgon_norm_reg_weight,
            balanced_train_loader = args.balanced_train_loader,
            # do_polygon_random_start = args.do_polygon_random_start
            )

    args_combine = "/{args_data:s}-{model_type:s}-{args_train:s}".format(
            args_data = args_data,
            model_type=args.model_type,
            args_train = args_train
            
            )
    return args_combine



class Trainer():
    """
    Trainer
    """
    def __init__(self, args, img_gdf, console = True):
        
 
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

        

        self.img_gdf = img_gdf
        
        args = update_args(args)
        self.args = args
        
        self.img_cla_dataset, self.img_cla_dataloader, self.split2num, self.data_split_col, self.train_sampler, self.num_classes = self.load_image_cla_dataset_dataloader(
        																																		img_gdf, 
																															                    num_worker = args.num_worker, 
																															                    batch_size = args.batch_size, 
																															                    balanced_train_loader = args.balanced_train_loader,
																															                    id_col = "ID", 
																															                    img_col = "IMAGE", 
																															                    class_col = "TYPEID", 
																															                    data_split_num = args.data_split_num, 
																															                    do_data_augment = args.do_data_augment, 
																															                    device = args.device)

        self.model = get_resnet_model(resnet_type = args.model_type, num_classes = self.num_classes).to(args.device)


        if args.opt == "sgd":
            self.optimizer = optim.SGD(filter(lambda p : p.requires_grad, self.model.parameters()), lr=args.lr, momentum=0)
        elif args.opt == "adam":
            self.optimizer = optim.Adam(filter(lambda p : p.requires_grad, self.model.parameters()), lr=args.lr, weight_decay = args.weight_decay)

        print("create model from {}".format(self.args_combine + ".pth"))
        self.logger.info("Save file at {}".format(self.args_combine + ".pth"))

        self.tb_writer = SummaryWriter(self.tb_log_dir)
        self.global_batch_idx = 0

    def load_image_cla_dataset_dataloader(self, img_gdf, 
                    num_worker, batch_size, balanced_train_loader,
                    id_col = "ID", img_col = "IMAGE", class_col = "TYPEID", data_split_num = 0, 
                    do_data_augment = False, device = "cpu"):
        '''
        load polygon classification dataset including training, validation, testing
        '''
        img_cla_dataset  = dict()
        img_cla_dataloader = dict()
        data_split_col = "SPLIT_{:d}".format(data_split_num)
        split2num = {"TRAIN": 1, "TEST": 0, "VALID": -1}
        split_nums = np.unique(np.array(img_gdf[data_split_col]))
        assert 1 in split_nums and 0 in split_nums
        dup_test = False
        if -1 not in split_nums:
            # we will make valid and test the same
            dup_test = True

        un_class = np.unique(np.array(img_gdf[class_col]))
        num_classes = len(un_class)
        max_num_exs_per_class = math.ceil(batch_size/num_classes)


        for split in ["TRAIN", "TEST", "VALID"]:
            # make dataset

            if split == "VALID" and dup_test:
                img_cla_dataset[split] = img_cla_dataset["TEST"]
            else:
                img_split_gdf = img_gdf[ img_gdf[data_split_col] == split2num[split] ]

                if split == "TRAIN": 
                    img_cla_dataset[split] = ImageDataset(img_gdf = img_split_gdf, 
                                                            id_col = id_col, 
                                                            img_col = img_col,
                                                            class_col = class_col,
                                                            do_data_augment = do_data_augment, 
                                                            device = device)
                else:
                    img_cla_dataset[split] = ImageDataset(img_gdf = img_split_gdf, 
                                                            id_col = id_col, 
                                                            img_col = img_col,
                                                            class_col = class_col,
                                                            do_data_augment = False, 
                                                            device = device)

            # make dataloader
            if split == "TRAIN":
                if balanced_train_loader:
                    train_sampler = BalancedSampler(classes = img_cla_dataset["TRAIN"].class_list.cpu().numpy(), 
                                                num_per_class = max_num_exs_per_class, 
                                                use_replace=False, 
                                                multi_label=False)
                    img_cla_dataloader[split] = torch.utils.data.DataLoader(img_cla_dataset[split], 
                                                            num_workers = num_worker, 
                                                            batch_size = batch_size,
                                                            sampler=train_sampler, 
                                                            shuffle = False)
                else:
                    train_sampler = None
                    img_cla_dataloader[split] = torch.utils.data.DataLoader(img_cla_dataset[split], 
                                                            num_workers = num_worker, 
                                                            batch_size = batch_size, 
                                                            shuffle = True)
            elif split == "VALID" and dup_test:
                img_cla_dataloader[split] = img_cla_dataloader['TEST']
            else:
                img_cla_dataloader[split] = torch.utils.data.DataLoader(img_cla_dataset[split], 
                                                            num_workers = num_worker, 
                                                            batch_size = batch_size,
                                                            shuffle = False)


        return img_cla_dataset, img_cla_dataloader, split2num, data_split_col, train_sampler, num_classes

    def run_train(self):
        # assert "norm" in self.geom_type_list
        self.global_batch_idx = train_image_model(
        							self.args, 
        							model = self.model, 
        							img_cla_dataloader = self.img_cla_dataloader, 
            						optimizer = self.optimizer, 
            						tb_writer = self.tb_writer, 
            						logger = self.logger, 
            						model_file = self.model_file, 
            						cla_epoches = self.args.cla_epoches, 
            						log_every = self.args.log_every, 
            						val_every = self.args.val_every, 
            						global_batch_idx = self.global_batch_idx)

    def load_model(self):
        self.model, self.optimizer, self.args = load_model(self.model, self.optimizer, self.model_file)