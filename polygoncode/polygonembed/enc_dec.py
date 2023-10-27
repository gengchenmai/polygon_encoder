import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from polygonembed.utils import *
from polygonembed.PolygonEncoder import *
from polygonembed.PolygonDecoder import *
from polygonembed.lenet import *
from polygonembed.resnet2d import *
from polygonembed.module import *


def get_img_cla_model(img_cla_type, num_classes = 20, in_channels = 1, 
        freqXY = [32, 32], hidden_dim = 512, mean = 0, std = 1, device = "cpu"):
    if "resnet" in img_cla_type:
        img_classifier = get_resnet_model(img_cla_type, num_classes = num_classes, in_channels = in_channels).to(device)
    elif img_cla_type == "lenet5":
        '''
        relation classification:
            in_channels = 2, subject + object
        Polygon classification:
            in_channels = 1, polygon 
        '''
        img_classifier = LeNet5(in_channels = in_channels, 
                                num_classes = num_classes,
                                signal_sizes = freqXY, # the height & width of polygon embedding
                                hidden_dim = hidden_dim,
                                mean = mean, 
                                std = std).to(device)

    return img_classifier


class PolygonEncoderDecoder(nn.Module):
    '''
    POlygon Encoder-Decoder to do unsupervised learning
    '''
    def __init__(self, pgon_enc, pgon_dec, loss_func = "NN", device = "cpu"):
        """
        Args:
            pgon_embed_dim: the output polygon embedding dimention
            loss_func: The generative loss function:
                L2: L2 distance between two corresponding points
                NN: nearest neighbor loss as Equation 1 in https://arxiv.org/pdf/1712.07262.pdf
                LOOPL2: Loop L2 distance
            device:
        """
        super(PolygonEncoderDecoder, self).__init__()
        self.pgon_enc = pgon_enc
        self.pgon_dec = pgon_dec
        self.pgon_embed_dim = pgon_dec.pgon_embed_dim
        self.num_vert = pgon_dec.num_vert
        self.loss_func = loss_func
        self.device = device

        if loss_func == "LOOPL2":
            self.loopl2_mask = make_loop_l2_mask(self.num_vert)


    def forward(self, polygons, do_polygon_random_start = False, add_first_pt = False):
        """
        Args:
            polygons: torch.FolarTensor(), shape (batch_size, num_vert+1, coord_dim = 2) 
                note that in num_vert dimention, the last point is the same as the 1st point
        Return:
            polygons_seq: torch.FolarTensor(), the input polygons
            polygons_seq_grt: torch.FolarTensor(), this generated polygons
                if add_first_pt:
                    shape (batch_size, num_vert+1, coord_dim = 2) 
                else:
                    shape (batch_size, num_vert, coord_dim = 2) 
            
        """
        # get rid of the last point to avoid confusing the circular padding
        # polygons_seq: shape (batch_size, num_vert, coord_dim = 2)  
        polygons_seq = polygons[:, :-1, :]

        batch_size, num_vert, coord_dim = polygons_seq.shape

        if do_polygon_random_start:
            polygons_seq = random_start_polygon_coords(polygons_seq)

        # pgon_embeds: shape (batch_size, pgon_embed_dim) 
        pgon_embeds = self.pgon_enc(polygons_seq)
        # polygons_seq_grt:  shape (batch_size, num_vert, coord_dim = 2) 
        # rand_grid: shape (batch_size, num_vert, coord_dim = 2) 
        polygons_seq_grt, rand_grid = self.pgon_dec(pgon_embeds)

        if add_first_pt:
            polygons_seq = add_first_pt_to_polygons(polygons_seq)
            polygons_seq_grt = add_first_pt_to_polygons(polygons_seq_grt)

        return polygons_seq, polygons_seq_grt, rand_grid
        

    def generative_loss(self, polygons, do_polygon_random_start = False):
        """
        Args:
            polygons: torch.FolarTensor(), shape (batch_size, num_vert+1, coord_dim = 2) 
                note that in num_vert dimention, the last point is the same as the 1st point
        Return:
            
        """
        # polygons_seq_grt: shape (batch_size, num_vert, coord_dim = 2) 
        polygons_seq, polygons_seq_grt, _ = self.forward(polygons, do_polygon_random_start, add_first_pt = False)

        if self.loss_func == "L2":
            # batch_loss: mean(deltax^2 + deltay^2), shape (batch_size)
            batch_loss = self.compute_l2_loss(polygons_seq, polygons_seq_grt)
        elif self.loss_func == "NN":
            # batch_loss: nearest neighbor loss, shape (batch_size)
            batch_loss = self.compute_nearest_neighbor_loss(polygons_seq, polygons_seq_grt)
        elif self.loss_func == "LOOPL2":
            # batch_loss: loop l2 loss, shape (batch_size)
            batch_loss = self.compute_loop_l2_loss(polygons_seq, polygons_seq_grt)
        return torch.mean(batch_loss)


    def compute_l2_loss(self, polygons_seq, polygons_seq_grt):
        '''
        Args:
            polygons_seq: torch.FolarTensor(), the input polygons, shape (batch_size, num_vert, coord_dim = 2) 
            polygons_seq_grt: torch.FolarTensor(), this generated polygons, shape (batch_size, num_vert, coord_dim = 2) 
        Return:
            l2_dist_mean: mean of the L2 distance between each pair of polygons - mean(deltax^2 + deltay^2), shape (batch_size)
        '''
        # l2_dist: deltax^2 + deltay^2, shape (batch_size, num_vert)
        l2_dist = torch.sum(torch.square(polygons_seq - polygons_seq_grt), dim = 2, keepdim = False)

        # l2_dist_mean: mean(deltax^2 + deltay^2), shape (batch_size)
        l2_dist_mean = torch.mean(l2_dist, dim = 1, keepdim = False)
        return l2_dist_mean

    def compute_nearest_neighbor_loss(self, polygons_seq, polygons_seq_grt):
        '''
        Args:
            polygons_seq: torch.FolarTensor(), the input polygons, shape (batch_size, num_vert, coord_dim = 2) 
            polygons_seq_grt: torch.FolarTensor(), this generated polygons, shape (batch_size, num_vert, coord_dim = 2) 
        Return:
            l2_dist_mean: mean of the L2 distance between each pair of polygons - mean(deltax^2 + deltay^2), shape (batch_size)
        '''
        nn_loss = batch_nearest_neighbor_loss(polygons_seq, polygons_seq_grt)
        return nn_loss

    def compute_loop_l2_loss(self, polygons_seq, polygons_seq_grt):
        '''
        for each polygon pair (P, Q), Given a specific k, where k = 0,1,2,..num_vert-1
        we compute the sum of the distance between P[i] and Q[i+k]
        loop_l2_loss is the min distance between (P, Q)
        Args:
            polygons_seq: torch.FolarTensor(), the input polygons, shape (batch_size, num_vert, coord_dim = 2) 
            polygons_seq_grt: torch.FolarTensor(), this generated polygons, shape (batch_size, num_vert, coord_dim = 2) 
        Return:
            loop_l2_loss: shape (batch_size), the min of the sum of distance between (P, Q)
        '''
        loop_l2_loss = batch_loop_l2_loss(polygons_seq, polygons_seq_grt, 
                                    loop_l2_mask = self.loopl2_mask, device = self.device)
        return loop_l2_loss





class PolygonClassifer(nn.Module):
    '''
    Polygon Classifer to do polygon classification
    '''
    def __init__(self, pgon_enc, num_classes = 20, pgon_norm_reg_weight = 0.1, do_weight_norm = False, device = "cpu"):
        """
        Args:
            pgon_embed_dim: the output polygon embedding dimention
            loss_func: The generative loss function:
                L2: L2 distance between two corresponding points
                NN: nearest neighbor loss as Equation 1 in https://arxiv.org/pdf/1712.07262.pdf
                LOOPL2: Loop L2 distance
            device:
        """
        super(PolygonClassifer, self).__init__()
        self.pgon_enc = pgon_enc
        self.pgon_embed_dim = pgon_enc.pgon_embed_dim
        # self.num_vert = pgon_dec.num_vert
        self.num_classes = num_classes
        self.device = device
        self.pgon_norm_reg_weight = pgon_norm_reg_weight

        

        
        if do_weight_norm:
            self.class_emb = nn.utils.weight_norm(module = nn.Linear(self.pgon_embed_dim, num_classes, bias=False), 
                                        name = 'weight', 
                                        dim = 1)
        else:
            self.class_emb = nn.Linear(self.pgon_embed_dim, num_classes, bias=True)
        nn.init.xavier_uniform_(self.class_emb.weight)
        nn.init.zeros_(self.class_emb.bias)
        

        self.softmax = nn.Softmax(dim=1)

        self.celoss = nn.CrossEntropyLoss()

    def forward(self, polygons, labels, do_polygon_random_start = False,
            V = None, E = None):
        """
        Args:
            polygons: torch.FolarTensor(), shape (batch_size, num_vert+1, coord_dim = 2) 
                note that in num_vert dimention, the last point is the same as the 1st point
            labels: torch.LongTensor(), shape (batch_size), the class of each polygon
            V: FloatTensor, [batch_size, num_vert, 2], vertice matrix
            E: LongTensor, [batch_size, num_vert, 2], vertice connection, edge matrix
        Return:
            cla_loss: classification loss, crossentropy
            embed_norm_loss, polygon embeding L2 norm loss
            
        """
        # class_pred: the prediction for each class, shape (batch_size, num_classes)
        # pgon_embeds: shape (batch_size, pgon_embed_dim)
        # Note that 
        class_pred, pgon_embeds = self.compute_class_pred(polygons, do_polygon_random_start, 
                                                        do_softmax = False,
                                                        V = V,
                                                        E = E)

        cla_loss = self.celoss(class_pred, labels)

        # pgon_embeds_norm: shape (batch_size)
        pgon_embeds_norm = torch.norm(pgon_embeds, p = 2, dim = 1, keepdim = False)

        pgon_embeds_norm = torch.mean(pgon_embeds_norm)

        loss = cla_loss + self.pgon_norm_reg_weight * pgon_embeds_norm


        return loss

    def compute_polygon_embed(self, polygons, do_polygon_random_start = False, V = None, E = None):
        """
        Args:
            polygons: torch.FolarTensor(), shape (batch_size, num_vert+1, coord_dim = 2) 
                note that in num_vert dimention, the last point is the same as the 1st point
            V: FloatTensor, [batch_size, num_vert, 2], vertice matrix
            E: LongTensor, [batch_size, num_vert, 2], vertice connection, edge matrix
        Return:
            pgon_embeds: shape (batch_size, pgon_embed_dim)
            
        """
        if V is not None and E is not None:
            # pgon_embeds: shape (batch_size, pgon_embed_dim) 
            pgon_embeds = self.pgon_enc(polygons = None, V = V, E = E)
        else:
            # get rid of the last point to avoid confusing the circular padding
            # polygons_seq: shape (batch_size, num_vert, coord_dim = 2)  
            polygons_seq = polygons[:, :-1, :]

            batch_size, num_vert, coord_dim = polygons_seq.shape

            if do_polygon_random_start:
                polygons_seq = random_start_polygon_coords(polygons_seq)

            # pgon_embeds: shape (batch_size, pgon_embed_dim) 
            pgon_embeds = self.pgon_enc(polygons_seq)
        return pgon_embeds

    def compute_class_pred(self, polygons, do_polygon_random_start = False, do_softmax = False,
                    V = None, E = None):
        """
        Args:
            polygons: torch.FolarTensor(), shape (batch_size, num_vert+1, coord_dim = 2) 
                note that in num_vert dimention, the last point is the same as the 1st point

            V: FloatTensor, [batch_size, num_vert, 2], vertice matrix
            E: LongTensor, [batch_size, num_vert, 2], vertice connection, edge matrix
        Return:
            class_pred: the prediction for each class, shape (batch_size, num_classes)
            pgon_embeds: the polygon embedding, shape (batch_size, pgon_embed_dim)
        """
        pgon_embeds = self.compute_polygon_embed(polygons, do_polygon_random_start = do_polygon_random_start,
                        V = V, E = E)

        # class_pred: shape (batch_size, num_classes)
        class_pred = self.class_emb(pgon_embeds)

        if do_softmax:
            class_pred = self.softmax(class_pred)
        return class_pred, pgon_embeds


class PolygonImageClassifer(nn.Module):
    '''
    Polygon Classifer to do polygon classification
    '''
    def __init__(self, pgon_enc, num_classes = 20, pgon_norm_reg_weight = 0.1, hidden_dim = 250, 
        mean = 0, std = 1, img_cla_type = "lenet5", device = "cpu"):
        """
        Args:
            pgon_enc: polygon encoder, NUFTIFFTPolygonEncoder()
            num_classes: number of class/relation need to predict
            pgon_norm_reg_weight:
            mean, std, the mean and stdev of norm used in LeNet5()
            device:
        """
        super(PolygonImageClassifer, self).__init__()
        self.pgon_enc = pgon_enc
        self.freqXY = pgon_enc.freqXY
        # self.num_vert = pgon_dec.num_vert
        self.num_classes = num_classes
        self.device = device
        self.pgon_norm_reg_weight = pgon_norm_reg_weight

        self.hidden_dim = hidden_dim
        self.mean = mean
        self.std = std

        self.img_cla_type = img_cla_type
        self.classifier = get_img_cla_model(
                                img_cla_type, 
                                num_classes = num_classes, 
                                in_channels = 1, # polygon, 1 channels
                                freqXY = self.freqXY, # the height & width of polygon embedding
                                hidden_dim = self.hidden_dim,
                                mean = mean, 
                                std = std, 
                                device = device)

        # self.classifier = LeNet5(in_channels = 1, # polygon, 1 channels
        #                         num_classes = num_classes,
        #                         signal_sizes = self.freqXY, # the height & width of polygon embedding
        #                         hidden_dim = self.hidden_dim,
        #                         mean = mean, 
        #                         std = std).to(device)

        self.softmax = nn.Softmax(dim=1)

        self.celoss = nn.CrossEntropyLoss()


    def forward(self, polygons, labels, do_polygon_random_start = False,
            V = None, E = None):
        """
        Args:
            polygons: torch.FolarTensor(), shape (batch_size, num_vert+1, coord_dim = 2) 
                note that in num_vert dimention, the last point is the same as the 1st point
            labels: torch.LongTensor(), shape (batch_size), the class of each polygon
            V: FloatTensor, [batch_size, num_vert, 2], vertice matrix
            E: LongTensor, [batch_size, num_vert, 2], vertice connection, edge matrix
        Return:
            cla_loss: classification loss, crossentropy
            embed_norm_loss, polygon embeding L2 norm loss
            
        """
        # class_pred: the prediction for each class, shape (batch_size, num_classes)
        # pgon_embeds: shape (batch_size, fx, fy, n_channel = 1)
        # Note that 
        class_pred, pgon_embeds = self.compute_class_pred(polygons, do_polygon_random_start, 
                                                        do_softmax = False,
                                                        V = V,
                                                        E = E)

        cla_loss = self.celoss(class_pred, labels)

        # # pgon_embeds_norm: shape (batch_size)
        # pgon_embeds_norm = torch.norm(pgon_embeds, p = 2, dim = 1, keepdim = False)

        # pgon_embeds_norm = torch.mean(pgon_embeds_norm)

        # loss = cla_loss + self.pgon_norm_reg_weight * pgon_embeds_norm
        loss = cla_loss


        return loss

    def compute_polygon_embed(self, polygons, do_polygon_random_start = False,
            V = None, E = None):
        """
        Args:
            polygons: torch.FolarTensor(), shape (batch_size, num_vert+1, coord_dim = 2) 
                note that in num_vert dimention, the last point is the same as the 1st point
            V: FloatTensor, [batch_size, num_vert, 2], vertice matrix
            E: LongTensor, [batch_size, num_vert, 2], vertice connection, edge matrix
        Return:
            pgon_embeds: shape (batch_size, fx, fy, n_channel = 1)
            
        """
        if V is not None and E is not None:
            # pgon_embeds: shape (batch_size, fx, fy, n_channel = 1)
            pgon_embeds = self.pgon_enc(polygons = None, V = V, E = E)
        else:
            # get rid of the last point to avoid confusing the circular padding
            # polygons_seq: shape (batch_size, num_vert, coord_dim = 2)  
            polygons_seq = polygons[:, :-1, :]

            batch_size, num_vert, coord_dim = polygons_seq.shape

            if do_polygon_random_start:
                polygons_seq = random_start_polygon_coords(polygons_seq)

            # pgon_embeds: shape (batch_size, fx, fy, n_channel = 1)
            pgon_embeds = self.pgon_enc(polygons_seq)
            
        _, fx, fy, n_channel = pgon_embeds.shape
        
        assert n_channel == 1
        assert fx == self.freqXY[0]
        assert fy == self.freqXY[1]
        return pgon_embeds

    def compute_class_pred(self, polygons, do_polygon_random_start = False, do_softmax = False,
                V = None, E = None):
        """
        Args:
            polygons: torch.FolarTensor(), shape (batch_size, num_vert+1, coord_dim = 2) 
                note that in num_vert dimention, the last point is the same as the 1st point
            V: FloatTensor, [batch_size, num_vert, 2], vertice matrix
            E: LongTensor, [batch_size, num_vert, 2], vertice connection, edge matrix
        Return:
            class_pred: the prediction for each class, shape (batch_size, num_classes)
            pgon_embeds: the polygon embedding, shape (batch_size, pgon_embed_dim)
        """
        # pgon_embeds: shape (batch_size, fx, fy, n_channel = 1)
        pgon_embeds = self.compute_polygon_embed(polygons, 
                                do_polygon_random_start = do_polygon_random_start, 
                                V = V, E = E)

        # pgon_embeds: shape (batch_size, 1, fx, fy)
        pgon_embeds = pgon_embeds.permute(0, 3, 1, 2)

        # class_pred: shape (batch_size, num_classes)
        class_pred = self.classifier(pgon_embeds)

        if do_softmax:
            class_pred = self.softmax(class_pred)
        return class_pred, pgon_embeds








class RelationConcatClassifer(nn.Module):
    '''
    spatial relation classifer to do relation classification
    '''
    def __init__(self, pgon_enc, num_classes = 10, pgon_norm_reg_weight = 0.1, do_weight_norm = False, device = "cpu"):
        """
        Args:
            pgon_enc: polygon encoder
            num_classes: number of class/relation need to predict
            pgon_norm_reg_weight:
            do_weight_norm:
            device:
        """
        super(RelationConcatClassifer, self).__init__()
        self.pgon_enc = pgon_enc
        self.pgon_embed_dim = pgon_enc.pgon_embed_dim
        # self.num_vert = pgon_dec.num_vert
        self.num_classes = num_classes
        self.device = device
        self.pgon_norm_reg_weight = pgon_norm_reg_weight

        

        
        if do_weight_norm:
            self.class_emb = nn.utils.weight_norm(module = nn.Linear(self.pgon_embed_dim * 2, num_classes, bias=False), 
                                        name = 'weight', 
                                        dim = 1)
        else:
            self.class_emb = nn.Linear(self.pgon_embed_dim * 2, num_classes, bias=True)
        nn.init.xavier_uniform_(self.class_emb.weight)
        nn.init.zeros_(self.class_emb.bias)
        

        self.softmax = nn.Softmax(dim=1)

        self.celoss = nn.CrossEntropyLoss()

    def forward(self, sub_pgons, obj_pgons, labels, do_polygon_random_start = False, 
                    sub_Vs = None, sub_Es = None, obj_Vs = None, obj_Es = None):
        """
        Args:
            sub_pgons, obj_pgons: torch.FolarTensor(), shape (batch_size, num_vert+1, coord_dim = 2) 
                note that in num_vert dimention, the last point is the same as the 1st point
            labels: torch.LongTensor(), shape (batch_size), the class of each polygon
            sub_Vs, obj_Vs: float tensor, [batch_size, num_vert, 2]
            sub_Es, obj_Es: int tensor, [batch_size, num_vert, 2]
            
        Return:
            cla_loss: classification loss, crossentropy
            embed_norm_loss, polygon embeding L2 norm loss
            
        """
        # class_pred: the prediction for each class, shape (batch_size, num_classes)
        # pgon_embeds: shape (batch_size, pgon_embed_dim)
        # Note that 
        class_pred, sub_pgon_embeds, obj_pgon_embeds = self.compute_class_pred(sub_pgons, obj_pgons, do_polygon_random_start, do_softmax = False,
                                sub_Vs = sub_Vs, sub_Es = sub_Es, obj_Vs = obj_Vs, obj_Es = obj_Es)

        cla_loss = self.celoss(class_pred, labels)

        # sub_pgon_embeds_norm: shape (batch_size)
        sub_pgon_embeds_norm = torch.norm(sub_pgon_embeds, p = 2, dim = 1, keepdim = False)

        sub_pgon_embeds_norm = torch.mean(sub_pgon_embeds_norm)

        # obj_pgon_embeds_norm: shape (batch_size)
        obj_pgon_embeds_norm = torch.norm(obj_pgon_embeds, p = 2, dim = 1, keepdim = False)
        
        obj_pgon_embeds_norm = torch.mean(obj_pgon_embeds_norm)

        loss = cla_loss + self.pgon_norm_reg_weight * (sub_pgon_embeds_norm + obj_pgon_embeds_norm)


        return loss

    def compute_polygon_embed(self, polygons, do_polygon_random_start = False, V = None, E = None):
        """
        Args:
            polygons: torch.FolarTensor(), shape (batch_size, num_vert+1, coord_dim = 2) 
                note that in num_vert dimention, the last point is the same as the 1st point
            V: float tensor, [batch_size, num_vert, 2], coordinate, x,y in [-1,1, -1, 1]
            E: int tensor, [batch_size, num_vert, 2], connectivity
        Return:
            pgon_embeds: shape (batch_size, pgon_embed_dim)
            
        """
        if V is not None and E is not None:
            # pgon_embeds: shape (batch_size, pgon_embed_dim) 
            pgon_embeds = self.pgon_enc(polygons = None, V = V, E = E)
        else:
            # get rid of the last point to avoid confusing the circular padding
            # polygons_seq: shape (batch_size, num_vert, coord_dim = 2)  
            polygons_seq = polygons[:, :-1, :]

            batch_size, num_vert, coord_dim = polygons_seq.shape

            if do_polygon_random_start:
                polygons_seq = random_start_polygon_coords(polygons_seq)

            # pgon_embeds: shape (batch_size, pgon_embed_dim) 
            pgon_embeds = self.pgon_enc(polygons_seq)
        return pgon_embeds

    def compute_class_pred(self, sub_pgons, obj_pgons, do_polygon_random_start = False, do_softmax = False,
                    sub_Vs = None, sub_Es = None, obj_Vs = None, obj_Es = None):
        """
        Args:
            sub_pgons, obj_pgons: torch.FolarTensor(), shape (batch_size, num_vert+1, coord_dim = 2) 
                note that in num_vert dimention, the last point is the same as the 1st point
            sub_Vs, obj_Vs: float tensor, [batch_size, num_vert, 2]
            sub_Es, obj_Es: int tensor, [batch_size, num_vert, 2]
        Return:
            class_pred: the prediction for each class, shape (batch_size, num_classes)
            pgon_embeds: the polygon embedding, shape (batch_size, pgon_embed_dim)
        """
        # sub_pgon_embeds, obj_pgon_embeds: shape (batch_size, pgon_embed_dim)
        sub_pgon_embeds = self.compute_polygon_embed(sub_pgons, do_polygon_random_start = False,
                            V = sub_Vs, E = sub_Es)

        obj_pgon_embeds = self.compute_polygon_embed(obj_pgons, do_polygon_random_start = False,
                            V = obj_Vs, E = obj_Es)

        # pgon_embeds: shape (batch_size, pgon_embed_dim * 2)
        pgon_embeds = torch.cat([sub_pgon_embeds, obj_pgon_embeds], dim = -1)

        # class_pred: shape (batch_size, num_classes)
        class_pred = self.class_emb(pgon_embeds)

        if do_softmax:
            class_pred = self.softmax(class_pred)
        return class_pred, sub_pgon_embeds, obj_pgon_embeds







class RelationImageConcatClassifer(nn.Module):
    '''
    spatial relation classifer to do relation classification
    Here, subject and object polygon are represented as 2D embedding (fx, fy, 1), 
    concatenate them together, we have (fx, fy, 2)
    '''
    def __init__(self, pgon_enc, num_classes = 10, pgon_norm_reg_weight = 0.1, hidden_dim = 250,
            mean = 0, std = 1, img_cla_type = "lenet5", device = "cpu"):
        """
        Args:
            pgon_enc: polygon encoder, NUFTIFFTPolygonEncoder()
            num_classes: number of class/relation need to predict
            pgon_norm_reg_weight:
            mean, std, the mean and stdev of norm used in LeNet5()
            device:
        """
        super(RelationImageConcatClassifer, self).__init__()
        self.pgon_enc = pgon_enc
        self.freqXY = pgon_enc.freqXY
        # self.num_vert = pgon_dec.num_vert
        self.num_classes = num_classes
        self.device = device
        self.pgon_norm_reg_weight = pgon_norm_reg_weight

        self.hidden_dim = hidden_dim
        self.mean = mean
        self.std = std

        self.img_cla_type = img_cla_type
        self.rel_classifier = get_img_cla_model(
                                img_cla_type, 
                                num_classes = num_classes, 
                                in_channels = 2, 
                                freqXY = self.freqXY, # the height & width of polygon embedding
                                hidden_dim = self.hidden_dim,
                                mean = mean, 
                                std = std, 
                                device = device)

        # self.rel_classifier = LeNet5(in_channels = 2, # subject + object, 2 channels
        #                         num_classes = num_classes,
        #                         signal_sizes = self.freqXY, # the height & width of polygon embedding
        #                         hidden_dim = self.hidden_dim,
        #                         mean = mean, 
        #                         std = std).to(device)
        

        self.softmax = nn.Softmax(dim=1)

        self.celoss = nn.CrossEntropyLoss()

    def forward(self, sub_pgons, obj_pgons, labels, do_polygon_random_start = False,
                    sub_Vs = None, sub_Es = None, obj_Vs = None, obj_Es = None):
        """
        Args:
            sub_pgons, obj_pgons: torch.FolarTensor(), shape (batch_size, num_vert+1, coord_dim = 2) 
                note that in num_vert dimention, the last point is the same as the 1st point
            labels: torch.LongTensor(), shape (batch_size), the class of each polygon
            sub_Vs, obj_Vs: float tensor, [batch_size, num_vert, 2]
            sub_Es, obj_Es: int tensor, [batch_size, num_vert, 2]
        Return:
            cla_loss: classification loss, crossentropy
            embed_norm_loss, polygon embeding L2 norm loss
            
        """
        # class_pred: the prediction for each class, shape (batch_size, num_classes)
        # pgon_embeds: shape (batch_size, pgon_embed_dim)
        # Note that 
        class_pred, sub_pgon_embeds, obj_pgon_embeds = self.compute_class_pred(
                                                        sub_pgons, obj_pgons, 
                                                        do_polygon_random_start, do_softmax = False,
                                                        sub_Vs = sub_Vs, sub_Es = sub_Es, 
                                                        obj_Vs = obj_Vs, obj_Es = obj_Es)

        cla_loss = self.celoss(class_pred, labels)

        # # sub_pgon_embeds_norm: shape (batch_size)
        # sub_pgon_embeds_norm = torch.norm(sub_pgon_embeds, p = 2, dim = 1, keepdim = False)

        # sub_pgon_embeds_norm = torch.mean(sub_pgon_embeds_norm)

        # # obj_pgon_embeds_norm: shape (batch_size)
        # obj_pgon_embeds_norm = torch.norm(obj_pgon_embeds, p = 2, dim = 1, keepdim = False)
        
        # obj_pgon_embeds_norm = torch.mean(obj_pgon_embeds_norm)

        # loss = cla_loss + self.pgon_norm_reg_weight * (sub_pgon_embeds_norm + obj_pgon_embeds_norm)

        loss = cla_loss

        return loss

    def compute_polygon_embed(self, polygons, do_polygon_random_start = False, V = None, E = None):
        """
        Args:
            polygons: torch.FolarTensor(), shape (batch_size, num_vert+1, coord_dim = 2) 
                note that in num_vert dimention, the last point is the same as the 1st point
            V: float tensor, [batch_size, num_vert, 2], coordinate, x,y in [-1,1, -1, 1]
            E: int tensor, [batch_size, num_vert, 2], connectivity
        Return:
            pgon_embeds: shape (batch_size, fx, fy, n_channel = 1)
            
        """
        if V is not None and E is not None:
            # pgon_embeds: shape (batch_size, fx, fy, n_channel = 1)
            pgon_embeds = self.pgon_enc(polygons = None, V = V, E = E)
        else:
            # get rid of the last point to avoid confusing the circular padding
            # polygons_seq: shape (batch_size, num_vert, coord_dim = 2)  
            polygons_seq = polygons[:, :-1, :]

            batch_size, num_vert, coord_dim = polygons_seq.shape

            if do_polygon_random_start:
                polygons_seq = random_start_polygon_coords(polygons_seq)

            # pgon_embeds: shape (batch_size, fx, fy, n_channel = 1)
            pgon_embeds = self.pgon_enc(polygons_seq)

        _, fx, fy, n_channel = pgon_embeds.shape
        
        assert n_channel == 1
        assert fx == self.freqXY[0]
        assert fy == self.freqXY[1]
        return pgon_embeds

    def compute_class_pred(self, sub_pgons, obj_pgons, do_polygon_random_start = False, do_softmax = False,
                        sub_Vs = None, sub_Es = None, obj_Vs = None, obj_Es = None):
        """
        Args:
            sub_pgons, obj_pgons: torch.FolarTensor(), shape (batch_size, num_vert+1, coord_dim = 2) 
                note that in num_vert dimention, the last point is the same as the 1st point
            sub_Vs, obj_Vs: float tensor, [batch_size, num_vert, 2]
            sub_Es, obj_Es: int tensor, [batch_size, num_vert, 2]
        Return:
            class_pred: the prediction for each class, shape (batch_size, num_classes)
            pgon_embeds: the polygon embedding, shape (batch_size, pgon_embed_dim)
        """
        # sub_pgon_embeds, obj_pgon_embeds: shape (batch_size, fx, fy, n_channel = 1)
        sub_pgon_embeds = self.compute_polygon_embed(sub_pgons, do_polygon_random_start = False,
                            V = sub_Vs, E = sub_Es)

        obj_pgon_embeds = self.compute_polygon_embed(obj_pgons, do_polygon_random_start = False,
                            V = obj_Vs, E = obj_Es)

        # pgon_embeds: shape (batch_size, fx, fy, 2)
        pgon_embeds = torch.cat([sub_pgon_embeds, obj_pgon_embeds], dim = -1)

        # pgon_embeds: shape (batch_size, 2, fx, fy)
        pgon_embeds = pgon_embeds.permute(0, 3, 1, 2)

        # class_pred: shape (batch_size, num_classes)
        class_pred = self.rel_classifier(pgon_embeds)


        if do_softmax:
            class_pred = self.softmax(class_pred)
        return class_pred, sub_pgon_embeds, obj_pgon_embeds




class RelationImageConcatMLPClassifer(nn.Module):
    '''
    spatial relation classifer to do relation classification
    Here, subject and object polygon are represented as 2D embedding (fx, fy, 1), 
    concatenate them together, we have (fx, fy, 2)
    '''
    def __init__(self, pgon_enc, num_classes = 10, 
            pgon_norm_reg_weight = 0.1, do_weight_norm = False,
            num_hidden_layers = 1, 
            hidden_dim = 512, 
            dropout = 0.5,
            f_act = "relu",
            use_layn = True, 
            skip_connection = True,
            mean = 0, std = 1, device = "cpu"):
        """
        Args:
            pgon_enc: polygon encoder, NUFTIFFTPolygonEncoder()
            num_classes: number of class/relation need to predict
            pgon_norm_reg_weight:
            mean, std, the mean and stdev of norm used in LeNet5()
            device:
        """
        super(RelationImageConcatMLPClassifer, self).__init__()
        self.pgon_enc = pgon_enc
        self.freqXY = pgon_enc.freqXY
        # self.num_vert = pgon_dec.num_vert
        self.num_classes = num_classes
        self.device = device
        self.pgon_norm_reg_weight = pgon_norm_reg_weight

        self.hidden_dim = hidden_dim
        self.mean = mean
        self.std = std

        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.f_act = f_act
        self.use_layn = use_layn
        self.skip_connection = skip_connection


        self.ffn = MultiLayerFeedForwardNN(
                                input_dim = self.freqXY[0] * self.freqXY[1] * 2,
                                output_dim = hidden_dim,
                                num_hidden_layers = num_hidden_layers,
                                dropout_rate = dropout,
                                hidden_dim = hidden_dim,
                                activation = f_act,
                                use_layernormalize = use_layn,
                                skip_connection = skip_connection,
                                context_str = "RelationImageConcatMLPClassifer")

        if do_weight_norm:
            self.class_emb = nn.utils.weight_norm(module = nn.Linear(hidden_dim, num_classes, bias=False), 
                                        name = 'weight', 
                                        dim = 1)
        else:
            self.class_emb = nn.Linear(hidden_dim, num_classes, bias=True)
        nn.init.xavier_uniform_(self.class_emb.weight)
        nn.init.zeros_(self.class_emb.bias)
         

        self.softmax = nn.Softmax(dim=1)

        self.celoss = nn.CrossEntropyLoss()

    def forward(self, sub_pgons, obj_pgons, labels, do_polygon_random_start = False,
                    sub_Vs = None, sub_Es = None, obj_Vs = None, obj_Es = None):
        """
        Args:
            sub_pgons, obj_pgons: torch.FolarTensor(), shape (batch_size, num_vert+1, coord_dim = 2) 
                note that in num_vert dimention, the last point is the same as the 1st point
            labels: torch.LongTensor(), shape (batch_size), the class of each polygon
            sub_Vs, obj_Vs: float tensor, [batch_size, num_vert, 2]
            sub_Es, obj_Es: int tensor, [batch_size, num_vert, 2]
        Return:
            cla_loss: classification loss, crossentropy
            embed_norm_loss, polygon embeding L2 norm loss
            
        """
        # class_pred: the prediction for each class, shape (batch_size, num_classes)
        # pgon_embeds: shape (batch_size, pgon_embed_dim)
        # Note that 
        class_pred, sub_pgon_embeds, obj_pgon_embeds = self.compute_class_pred(
                                                        sub_pgons, obj_pgons, 
                                                        do_polygon_random_start, do_softmax = False,
                                                        sub_Vs = sub_Vs, sub_Es = sub_Es, 
                                                        obj_Vs = obj_Vs, obj_Es = obj_Es)

        cla_loss = self.celoss(class_pred, labels)

        # # sub_pgon_embeds_norm: shape (batch_size)
        # sub_pgon_embeds_norm = torch.norm(sub_pgon_embeds, p = 2, dim = 1, keepdim = False)

        # sub_pgon_embeds_norm = torch.mean(sub_pgon_embeds_norm)

        # # obj_pgon_embeds_norm: shape (batch_size)
        # obj_pgon_embeds_norm = torch.norm(obj_pgon_embeds, p = 2, dim = 1, keepdim = False)
        
        # obj_pgon_embeds_norm = torch.mean(obj_pgon_embeds_norm)

        # loss = cla_loss + self.pgon_norm_reg_weight * (sub_pgon_embeds_norm + obj_pgon_embeds_norm)

        loss = cla_loss

        return loss

    def compute_polygon_embed(self, polygons, do_polygon_random_start = False, V = None, E = None):
        """
        Args:
            polygons: torch.FolarTensor(), shape (batch_size, num_vert+1, coord_dim = 2) 
                note that in num_vert dimention, the last point is the same as the 1st point
            V: float tensor, [batch_size, num_vert, 2], coordinate, x,y in [-1,1, -1, 1]
            E: int tensor, [batch_size, num_vert, 2], connectivity
        Return:
            pgon_embeds: shape (batch_size, fx, fy, n_channel = 1)
            
        """
        if V is not None and E is not None:
            # pgon_embeds: shape (batch_size, fx, fy, n_channel = 1)
            pgon_embeds = self.pgon_enc(polygons = None, V = V, E = E)
        else:
            # get rid of the last point to avoid confusing the circular padding
            # polygons_seq: shape (batch_size, num_vert, coord_dim = 2)  
            polygons_seq = polygons[:, :-1, :]

            batch_size, num_vert, coord_dim = polygons_seq.shape

            if do_polygon_random_start:
                polygons_seq = random_start_polygon_coords(polygons_seq)

            # pgon_embeds: shape (batch_size, fx, fy, n_channel = 1)
            pgon_embeds = self.pgon_enc(polygons_seq)

        _, fx, fy, n_channel = pgon_embeds.shape
        
        assert n_channel == 1
        assert fx == self.freqXY[0]
        assert fy == self.freqXY[1]
        return pgon_embeds

    def compute_class_pred(self, sub_pgons, obj_pgons, do_polygon_random_start = False, do_softmax = False,
                        sub_Vs = None, sub_Es = None, obj_Vs = None, obj_Es = None):
        """
        Args:
            sub_pgons, obj_pgons: torch.FolarTensor(), shape (batch_size, num_vert+1, coord_dim = 2) 
                note that in num_vert dimention, the last point is the same as the 1st point
            sub_Vs, obj_Vs: float tensor, [batch_size, num_vert, 2]
            sub_Es, obj_Es: int tensor, [batch_size, num_vert, 2]
        Return:
            class_pred: the prediction for each class, shape (batch_size, num_classes)
            pgon_embeds: the polygon embedding, shape (batch_size, pgon_embed_dim)
        """
        # sub_pgon_embeds, obj_pgon_embeds: shape (batch_size, fx, fy, n_channel = 1)
        sub_pgon_embeds = self.compute_polygon_embed(sub_pgons, do_polygon_random_start = False,
                            V = sub_Vs, E = sub_Es)

        obj_pgon_embeds = self.compute_polygon_embed(obj_pgons, do_polygon_random_start = False,
                            V = obj_Vs, E = obj_Es)

        # pgon_embeds: shape (batch_size, fx, fy, 2)
        pgon_embeds = torch.cat([sub_pgon_embeds, obj_pgon_embeds], dim = -1)

        batch_size, _, _, _ = pgon_embeds.shape
        # pgon_embeds: shape (batch_size, fx * fy * 2)
        pgon_embeds = pgon_embeds.reshape(batch_size, -1)

        # hidden_embeds: shape (batch_size, hidden_dim)
        hidden_embeds = self.ffn(pgon_embeds)

        # class_pred: shape (batch_size, num_classes)
        class_pred = self.class_emb(hidden_embeds)


        if do_softmax:
            class_pred = self.softmax(class_pred)
        return class_pred, sub_pgon_embeds, obj_pgon_embeds