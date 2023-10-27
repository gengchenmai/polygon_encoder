import numpy as np
import torch
from torch.utils.data.sampler import Sampler

from polygonembed.utils import *


class PolygonDataset(torch.utils.data.Dataset):
    def __init__(self, pgon_gdf, id_col = "ID", geom_type = "origin", class_col = None, 
                    do_data_augment = False, num_augment = 8, device = "cpu"):
        '''
        Args:
            pgon_gdf: GeoDataFrame()
            geom_type: the type of geometry we sample
                origin: the original geometry
                norm: the normalized geometry by their spatial extent
            class_col: the column name indicates the polygon class
        '''
        self.pgon_gdf = pgon_gdf
        self.id_col = id_col

        self.geom_type = geom_type
        self._set_geometry_col(geom_type)
        self.device = device
        self.class_col = class_col

        pgon_id_list = np.array(pgon_gdf[id_col])
        if class_col is not None: 
            class_list = np.array(pgon_gdf[class_col])
        else:
            class_list  = None
        
        # pgon_list: shape (num_pgon, num_vert+1, 2)
        pgon_list = [np.expand_dims(np.array(pgon.exterior.coords), axis = 0) for pgon in list(pgon_gdf[self.geometry_col])]
        pgon_list = np.concatenate(pgon_list, axis = 0)

        if do_data_augment:
            pgon_id_list, pgon_list, class_list = self.polygon_data_argumentation(pgon_id_list = pgon_id_list, 
                                                                                pgon_list = pgon_list, 
                                                                                class_list = class_list, 
                                                                                batch_size = 200, 
                                                                                num_augment = num_augment)

        self.pgon_id_list = torch.LongTensor(pgon_id_list).to(self.device)
        self.pgon_list = torch.FloatTensor(pgon_list).to(self.device)
        if self.class_col is not None:
            self.class_list =  torch.LongTensor(class_list).to(self.device)


    def polygon_data_argumentation(self, pgon_id_list, pgon_list, class_list = None, 
                    batch_size = 200, num_augment = 8):
        assert pgon_id_list.shape[0] == pgon_list.shape[0]
        if class_list is not None:
            assert class_list.shape[0] == pgon_list.shape[0]

        if num_augment > 0:
            pgon_id_list_rot, pgon_list_rot, class_list_rot = rotate_polygon_batch(pgon_id_list, pgon_list, class_list, batch_size = 500, num_augment = num_augment)

        pgon_id_list_flip, pgon_list_flip, class_list_flip = flip_polygon_batch(pgon_id_list, pgon_list, class_list, batch_size = 500)

        if num_augment > 0:
            pgon_id_list = np.concatenate([pgon_id_list, pgon_id_list_rot, pgon_id_list_flip], axis = 0)
            pgon_list = np.concatenate([pgon_list, pgon_list_rot, pgon_list_flip], axis = 0)
            if class_list is not None:
                class_list = np.concatenate([class_list, class_list_rot, class_list_flip], axis = 0)
        else:
            pgon_id_list = np.concatenate([pgon_id_list, pgon_id_list_flip], axis = 0)
            pgon_list = np.concatenate([pgon_list, pgon_list_flip], axis = 0)
            if class_list is not None:
                class_list = np.concatenate([class_list, class_list_flip], axis = 0)
        return pgon_id_list, pgon_list, class_list


    def _set_geometry_col(self, geom_type):
        if geom_type == "origin":
            self.geometry_col = "geometry"
        elif geom_type == "norm":
            self.geometry_col = "geometry_norm"
        else:
            raise Exception("Unknown geom_type")

    def __len__(self):
        return len(self.pgon_id_list)

    def __getitem__(self, idx):
        '''
        Import Note: if we want to use multiprocessing (torch.utils.data.DataLoader(...num_works = 6))
        We need to return  the data tensor as a CPU tensor, and cast to CUDA during enumrating
        (https://discuss.pytorch.org/t/cuda-initialization-error-when-dataloader-with-cuda-tensor/43390/2)
        Args:
            idx: the idx of the polygon object
        Return:
            id: tensor, [1]
            pgon_tensor: tensor, shape [num_vert+1, 2]
            
        True shape: (Note, enumarte this Dataset with Dataloader with add batch_size as the 1st dim)
            id: tensor, [batch_size]
            pgon_tensor: tensor, shape [batch_size, num_vert+1, 2]
        '''
        pgon_id = self.pgon_id_list[idx]
        # pgon = self.pgon_gdf[self.geometry_col].iloc[idx]
        # # pgon_tensor: shape (num_vert+1, 2)
        # pgon_tensor = torch.FloatTensor(np.array(pgon.exterior.coords)).to(self.device)
        pgon_tensor = self.pgon_list[idx]
        if self.class_col is not None:
            label = self.class_list[idx]
            return pgon_id, pgon_tensor, label
        else:
            return pgon_id, pgon_tensor




class PolygonComplexDataset(torch.utils.data.Dataset):
    def __init__(self, pgon_gdf, id_col = "ID", geom_type = "origin", class_col = None, 
                    do_data_augment = False, num_augment = 8, device = "cpu"):
        '''
        Args:
            pgon_gdf: GeoDataFrame()
            geom_type: the type of geometry we sample
                origin: the original geometry
                norm: the normalized geometry by their spatial extent
            class_col: the column name indicates the polygon class
        '''
        self.pgon_gdf = pgon_gdf
        self.id_col = id_col

        self.geom_type = geom_type
        self._set_geometry_col(geom_type)
        self._set_V_E_col(geom_type)
        self.device = device
        self.class_col = class_col

        pgon_id_list = np.array(pgon_gdf[id_col])
        if class_col is not None: 
            class_list = np.array(pgon_gdf[class_col])
        else:
            class_list  = None

        
        
        

        if do_data_augment:
            raise Exception("Data augumentation not implemented for PolygonComplexDataset")
        #     pgon_id_list, pgon_list, class_list = self.polygon_data_argumentation(pgon_id_list = pgon_id_list, 
        #                                                                         pgon_list = pgon_list, 
        #                                                                         class_list = class_list, 
        #                                                                         batch_size = 200, 
        #                                                                         num_augment = num_augment)

        self.pgon_id_list = torch.LongTensor(pgon_id_list).to(self.device)
        self.V_list, self.E_list = self._get_V_E_list(V_col = self.V_col, E_col = self.E_col)
        if self.class_col is not None:
            self.class_list =  torch.LongTensor(class_list).to(self.device)

    def _get_V_E_list(self, V_col, E_col):
        '''
        Return:
            V: torch.FloatTensor, shape (num_pgon, num_vert, 2). vertex coordinates
            E: torch.LongTensor, shape (num_pgon, num_vert, 2). vertex connection
        '''
        
        V_list = list(self.pgon_gdf[V_col])
        # V: float, shape (num_pgon, num_vert, 2). vertex coordinates
        V = np.stack(V_list, axis = 0)
        V = torch.FloatTensor(V).to(self.device)

        E_list = list(self.pgon_gdf[E_col])
        # E: int, shape (num_pgon, num_vert, 2). vertex connection
        E = np.stack(E_list, axis = 0)
        E = torch.LongTensor(E).to(self.device)

        return V, E


    def _set_V_E_col(self, geom_type = "norm"):
        if geom_type == "norm":
            self.V_col = "V_norm"
            self.E_col = "E_norm"
        elif geom_type == "origin":
            self.V_col = "V"
            self.E_col = "E"
        else:
            raise Exception("Unknown task")


    def _set_geometry_col(self, geom_type):
        if geom_type == "origin":
            self.geometry_col = "geometry"
        elif geom_type == "norm":
            self.geometry_col = "geometry_norm"
        else:
            raise Exception("Unknown geom_type")

    def __len__(self):
        return len(self.pgon_id_list)

    def __getitem__(self, idx):
        '''
        Import Note: if we want to use multiprocessing (torch.utils.data.DataLoader(...num_works = 6))
        We need to return  the data tensor as a CPU tensor, and cast to CUDA during enumrating
        (https://discuss.pytorch.org/t/cuda-initialization-error-when-dataloader-with-cuda-tensor/43390/2)
        Args:
            idx: the idx of the polygon object
        Return:
            pgon_id: tensor, [1]
            V: torch.FloatTensor, shape (num_vert, 2). vertex coordinates
            E: torch.LongTensor, shape (num_vert, 2). vertex connection
            label: torch.LongTensor, shape [1]
            
        True shape: (Note, enumarte this Dataset with Dataloader with add batch_size as the 1st dim)
            id: tensor, [batch_size]
            pgon_tensor: tensor, shape [batch_size, num_vert+1, 2]
        '''
        pgon_id = self.pgon_id_list[idx]
        # pgon = self.pgon_gdf[self.geometry_col].iloc[idx]
        # # pgon_tensor: shape (num_vert+1, 2)
        # pgon_tensor = torch.FloatTensor(np.array(pgon.exterior.coords)).to(self.device)
        # pgon_tensor = self.pgon_list[idx]
        V = self.V_list[idx]
        E = self.E_list[idx]
        if self.class_col is not None:
            label = self.class_list[idx]
            return pgon_id, V, E, label
        else:
            return pgon_id, V, E


    # def polygon_data_argumentation(self, pgon_id_list, pgon_list, class_list = None, 
    #                 batch_size = 200, num_augment = 8):
    #     assert pgon_id_list.shape[0] == pgon_list.shape[0]
    #     if class_list is not None:
    #         assert class_list.shape[0] == pgon_list.shape[0]

    #     if num_augment > 0:
    #         pgon_id_list_rot, pgon_list_rot, class_list_rot = rotate_polygon_batch(pgon_id_list, pgon_list, class_list, batch_size = 500, num_augment = num_augment)

    #     pgon_id_list_flip, pgon_list_flip, class_list_flip = flip_polygon_batch(pgon_id_list, pgon_list, class_list, batch_size = 500)

    #     if num_augment > 0:
    #         pgon_id_list = np.concatenate([pgon_id_list, pgon_id_list_rot, pgon_id_list_flip], axis = 0)
    #         pgon_list = np.concatenate([pgon_list, pgon_list_rot, pgon_list_flip], axis = 0)
    #         if class_list is not None:
    #             class_list = np.concatenate([class_list, class_list_rot, class_list_flip], axis = 0)
    #     else:
    #         pgon_id_list = np.concatenate([pgon_id_list, pgon_id_list_flip], axis = 0)
    #         pgon_list = np.concatenate([pgon_list, pgon_list_flip], axis = 0)
    #         if class_list is not None:
    #             class_list = np.concatenate([class_list, class_list_flip], axis = 0)
    #     return pgon_id_list, pgon_list, class_list


        
class BalancedSampler(torch.utils.data.Sampler):
    # sample "evenly" from each from class
    def __init__(self, classes, num_per_class, use_replace=False, multi_label=False):
        '''
        Args:
            classes: 
                if multi_label == False: list(), [batch_size], the list of occurance type id (ground truth labels)
                if multi_label == True:  dict()
            num_per_class: the max number of sample per class
            use_replace: whether or not do sample with replacement
        '''
        self.class_dict = {}
        self.num_per_class = num_per_class
        self.use_replace = use_replace
        self.multi_label = multi_label

        if self.multi_label:
            self.class_dict = classes
        else:
            # standard classification
            un_classes = np.unique(classes)
            for cc in un_classes:
                self.class_dict[cc] = []

            for ii in range(len(classes)):
                self.class_dict[classes[ii]].append(ii)
            '''
            class_dict: dict()
                key: the class id
                value: a list of polygon sample index who belong to this class
            '''
        if self.use_replace:
            self.num_exs = self.num_per_class*len(un_classes)
        else:
            self.num_exs = 0
            for cc in self.class_dict.keys():
                self.num_exs += np.minimum(len(self.class_dict[cc]), self.num_per_class)


    def __iter__(self):
        indices = []
        for cc in self.class_dict:
            if self.use_replace:
                indices.extend(np.random.choice(self.class_dict[cc], self.num_per_class).tolist())
            else:
                indices.extend(np.random.choice(self.class_dict[cc], np.minimum(len(self.class_dict[cc]),
                                                self.num_per_class), replace=False).tolist())
        # in the multi label setting there will be duplictes at training time
        np.random.shuffle(indices)  # will remain a list
        return iter(indices)

    def __len__(self):
        return self.num_exs









class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_gdf, id_col = "ID", img_col = "IMAGE", class_col = "TYPEID", 
                    do_data_augment = False, device = "cpu"):
        '''
        Args:
            img_gdf: DataFrame()
            img_col: the column name indicate the image np matrox
            class_col: the column name indicates the polygon class
        '''
        self.img_gdf = img_gdf
        self.id_col = id_col
        self.img_col = img_col
        self.class_col = class_col

        self.device = device

        img_id_list = np.array(img_gdf[id_col])
        # image_list: shape (N, num_channels, height, width) => (1000, 1, 224, 224)
        img_list = np.concatenate(list(img_gdf[img_col]), axis = 0)
        class_list = np.array(img_gdf[class_col])
        
        
        # if do_data_augment:
        #     pass

        self.img_id_list = torch.LongTensor(img_id_list).to(self.device)
        self.img_list = torch.FloatTensor(img_list).to(self.device)
        self.class_list =  torch.LongTensor(class_list).to(self.device)


    
    def __len__(self):
        return len(self.img_id_list)

    def __getitem__(self, idx):
        '''
        Import Note: if we want to use multiprocessing (torch.utils.data.DataLoader(...num_works = 6))
        We need to return  the data tensor as a CPU tensor, and cast to CUDA during enumrating
        (https://discuss.pytorch.org/t/cuda-initialization-error-when-dataloader-with-cuda-tensor/43390/2)
        Args:
            idx: the idx of the polygon object
        Return:
            id: tensor, [1]
            img_tensor: tensor, shape (1, num_channels, height, width) => (1, 1, 224, 224)
            label: tensor, (1,)
            
        True shape: (Note, enumarte this Dataset with Dataloader with add batch_size as the 1st dim)
            img_id: tensor, [batch_size]
            img_tensor: tensor, shape (batch_size, num_channels, height, width) => (B, 1, 224, 224)
            label: tensor, (batch_size,)
        '''
        img_id = self.img_id_list[idx]
        img_tensor = self.img_list[idx]
        label = self.class_list[idx]
        return img_id, img_tensor, label
    





class TripleDataset(torch.utils.data.Dataset):
    def __init__(self, triple_gdf, pgon_gdf = None, task = "rel", id_col = "tripleID",
                     device = "cpu"):
        '''
        Args:
            triple_gdf: GeoDataFrame(), a list of triple whose subject and object are polygons
                tripleID: ID of each triple
                sid, oid: the entity index of subject and object
                rid: the id of spatial relation, the prediction target
                geom_s_norm, geom_o_norm: normalized subject and object polygons, 
                        they are normalized into X, Y in [-1,1] by using they shared bounding box
            task: the task we want to do
                rel: spatial relation prediction, need to use noralized geometry
                obj: object entity prediction (similar to link prediction), need to use original geometry
                sub: object entity prediction (similar to link prediction), need to use original geometry
            id_col: the triple iD
            

        '''
        self.triple_gdf = triple_gdf
        self.pgon_gdf = pgon_gdf
        self.id_col = id_col
        
        self.task = task
        self._set_geometry_col(task)
        self._set_class_col(task)

        self.device = device
        


        triple_id_list = np.array(triple_gdf[id_col])
        self.triple_id_list = torch.LongTensor(triple_id_list).to(self.device)


        class_list = np.array(triple_gdf[self.class_col])
        self.class_list =  torch.LongTensor(class_list).to(self.device)
        
        if task == "rel":
            self.sub_pgon_list = self._get_pgon_list(geometry_col = self.sub_geometry_col)
            self.obj_pgon_list = self._get_pgon_list(geometry_col = self.obj_geometry_col)
        else:
            raise Exception("Not Implementaed error")



        self.sid_list = torch.LongTensor(np.array(triple_gdf["sid"])).to(self.device)
        self.rid_list = torch.LongTensor(np.array(triple_gdf["rid"])).to(self.device)
        self.oid_list = torch.LongTensor(np.array(triple_gdf["oid"])).to(self.device)
    
    def get_num_classes(self):
        return len(np.unique(np.array(self.triple_gdf[self.class_col])))
            

    def _get_pgon_list(self, geometry_col):
        # pgon_list: shape (num_pgon, num_vert+1, 2)
        pgon_list = [np.expand_dims(np.array(pgon.exterior.coords), axis = 0) for pgon in list(self.triple_gdf[geometry_col])]
        pgon_list = np.concatenate(pgon_list, axis = 0)
        return torch.FloatTensor(pgon_list).to(self.device)


    def _set_geometry_col(self, task):
        if task == "sub" or task == "obj":
            raise Exception("Not Implementaed error")
        elif task == "rel":
            self.sub_geometry_col = "geom_s_norm"
            self.obj_geometry_col = "geom_o_norm"
        else:
            raise Exception("Unknown task")

    def _set_class_col(self, task):
        '''
        class_col: the column name indicates predicting label
                sid: when we do subject entity prediction, task = "sub"
                rid: when we do spatial relation prediction, task = "rel"
                oid: when we do object entity prediction, task = "obj"
        '''
        if task == "sub":
            self.class_col = "sid"
        elif task == "rel":
            self.class_col = "rid"
        elif task == "obj":
            self.class_col = "oid"
        else:
            raise Exception("Unknown task")

    def __len__(self):
        return len(self.triple_id_list)

    def __getitem__(self, idx):
        '''
        Import Note: if we want to use multiprocessing (torch.utils.data.DataLoader(...num_works = 6))
        We need to return  the data tensor as a CPU tensor, and cast to CUDA during enumrating
        (https://discuss.pytorch.org/t/cuda-initialization-error-when-dataloader-with-cuda-tensor/43390/2)
        Args:
            idx: the idx of the triple
        Return:
            triple_id: tensor, [1]
            sub_pgon: tensor, shape [num_vert+1, 2]
            obj_pgon: tensor, shape [num_vert+1, 2]
            label: tensor, [1], the target label, 
                task = "rel": the relation id
            
        True shape: (Note, enumarte this Dataset with Dataloader with add batch_size as the 1st dim)
            triple_id: tensor, [batch_size]
            sub_pgon: tensor, shape [batch_size, num_vert+1, 2]
            obj_pgon: tensor, shape [batch_size, num_vert+1, 2]
            label: tensor, [batch_size], the target label, 
                task = "rel": the relation id
        '''
        triple_id = self.triple_id_list[idx]

        sid = self.sid_list[idx]
        rid = self.rid_list[idx]
        oid = self.oid_list[idx]

        label = self.class_list[idx]
        if self.task == "rel":
            # sub_pgon, obj_pgon: shape (num_vert+1, 2)
            sub_pgon = self.sub_pgon_list[idx]
            obj_pgon = self.obj_pgon_list[idx]
            return triple_id, sid, rid, oid, sub_pgon, obj_pgon, label
        else:
            raise Exception("Not Implementaed error")


        
class TripleComplexDataset(torch.utils.data.Dataset):
    def __init__(self, triple_gdf, pgon_gdf = None, task = "rel", id_col = "tripleID",
                     device = "cpu"):
        '''
        Args:
            triple_gdf: GeoDataFrame(), a list of triple whose subject and object are polygons
                tripleID: ID of each triple
                sid, oid: the entity index of subject and object
                rid: the id of spatial relation, the prediction target
                V_s_norm, E_s_norm: V and E matrix of normalized subject polygons
                V_o_norm, E_o_norm: V and E matrix of normalized object polygons
                        they are normalized into X, Y in [-1,1] by using they shared bounding box
            task: the task we want to do
                rel: spatial relation prediction, need to use noralized geometry
                obj: object entity prediction (similar to link prediction), need to use original geometry
                sub: object entity prediction (similar to link prediction), need to use original geometry
            id_col: the triple iD
            

        '''
        self.triple_gdf = triple_gdf
        self.pgon_gdf = pgon_gdf
        self.id_col = id_col
        
        self.task = task
        self._set_V_E_col(task)
        self._set_class_col(task)

        self.device = device
        


        triple_id_list = np.array(triple_gdf[id_col])
        self.triple_id_list = torch.LongTensor(triple_id_list).to(self.device)


        class_list = np.array(triple_gdf[self.class_col])
        self.class_list =  torch.LongTensor(class_list).to(self.device)
        
        if task == "rel":
            self.sub_V_list, self.sub_E_list = self._get_V_E_list(
                V_col = self.sub_V_col, E_col = self.sub_E_col)
            self.obj_V_list, self.obj_E_list = self._get_V_E_list(
                V_col = self.obj_V_col, E_col = self.obj_E_col)
        else:
            raise Exception("Not Implementaed error")



        self.sid_list = torch.LongTensor(np.array(triple_gdf["sid"])).to(self.device)
        self.rid_list = torch.LongTensor(np.array(triple_gdf["rid"])).to(self.device)
        self.oid_list = torch.LongTensor(np.array(triple_gdf["oid"])).to(self.device)
    
    def get_num_classes(self):
        return len(np.unique(np.array(self.triple_gdf[self.class_col])))
            

    def _get_V_E_list(self, V_col, E_col):
        
        V_list = list(self.triple_gdf[V_col])
        # V: float, shape (num_pgon, num_vert, 2). vertex coordinates
        V = np.stack(V_list, axis = 0)
        V = torch.FloatTensor(V).to(self.device)

        E_list = list(self.triple_gdf[E_col])
        # E: int, shape (num_pgon, num_vert, 2). vertex connection
        E = np.stack(E_list, axis = 0)
        E = torch.LongTensor(E).to(self.device)

        return V, E


    def _set_V_E_col(self, task):
        if task == "sub" or task == "obj":
            raise Exception("Not Implementaed error")
        elif task == "rel":
            self.sub_V_col = "V_s_norm"
            self.sub_E_col = "E_s_norm"
            self.obj_V_col = "V_o_norm"
            self.obj_E_col = "E_o_norm"
        else:
            raise Exception("Unknown task")

    def _set_class_col(self, task):
        '''
        class_col: the column name indicates predicting label
                sid: when we do subject entity prediction, task = "sub"
                rid: when we do spatial relation prediction, task = "rel"
                oid: when we do object entity prediction, task = "obj"
        '''
        if task == "sub":
            self.class_col = "sid"
        elif task == "rel":
            self.class_col = "rid"
        elif task == "obj":
            self.class_col = "oid"
        else:
            raise Exception("Unknown task")

    def __len__(self):
        return len(self.triple_id_list)

    def __getitem__(self, idx):
        '''
        Import Note: if we want to use multiprocessing (torch.utils.data.DataLoader(...num_works = 6))
        We need to return  the data tensor as a CPU tensor, and cast to CUDA during enumrating
        (https://discuss.pytorch.org/t/cuda-initialization-error-when-dataloader-with-cuda-tensor/43390/2)
        Args:
            idx: the idx of the triple
        Return:
            triple_id: tensor, [1]
            sub_pgon: tensor, shape [num_vert+1, 2]
            obj_pgon: tensor, shape [num_vert+1, 2]
            label: tensor, [1], the target label, 
                task = "rel": the relation id
            
        True shape: (Note, enumarte this Dataset with Dataloader with add batch_size as the 1st dim)
            triple_id: tensor, [batch_size]
            sub_pgon: tensor, shape [batch_size, num_vert+1, 2]
            obj_pgon: tensor, shape [batch_size, num_vert+1, 2]
            label: tensor, [batch_size], the target label, 
                task = "rel": the relation id
        '''
        triple_id = self.triple_id_list[idx]

        sid = self.sid_list[idx]
        rid = self.rid_list[idx]
        oid = self.oid_list[idx]

        label = self.class_list[idx]
        if self.task == "rel":
            # sub_V, obj_V: shape (num_vert, 2)
            # sub_E, obj_E: shape (num_vert, 2)
            sub_V = self.sub_V_list[idx]
            sub_E = self.sub_E_list[idx]
            obj_V = self.obj_V_list[idx]
            obj_E = self.obj_E_list[idx]
            
            return triple_id, sid, rid, oid, sub_V, sub_E, obj_V, obj_E, label
        else:
            raise Exception("Not Implementaed error")