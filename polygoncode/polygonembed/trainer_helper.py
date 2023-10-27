from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from polygonembed.data_util import *
from polygonembed.PolygonEncoder import *
from polygonembed.PolygonDecoder import *
from polygonembed.enc_dec import *

def check_conv(vals, window=2, tol=1e-6):
    '''
    Check the convergence of mode based on the evaluation score:
    Args:
        vals: a list of evaluation score
        window: the average window size
        tol: the threshold for convergence
    '''
    if len(vals) < 2 * window:
        return False
    conv = np.mean(vals[-window:]) - np.mean(vals[-2*window:-window]) 
    return conv < tol

def update_loss(loss, losses, ema_loss, ema_alpha=0.01):
    losses.append(loss)
    if ema_loss is None:
        ema_loss = loss
    else:
        ema_loss = (1-ema_alpha)*ema_loss + ema_alpha*loss
    return losses, ema_loss

def save_model(model, optimizer, global_batch_idx, args, model_file):
    op_state = {'epoch': global_batch_idx,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'args' : args}
    torch.save(op_state, model_file)

def load_model(model, optimizer, model_file, device = 'cuda:0'):
    op_state = torch.load(model_file, map_location = device)
    args = op_state['args']

    model.load_state_dict(op_state['state_dict'])
    optimizer.load_state_dict(op_state['optimizer'])
    return model, optimizer, args


def train_model(args, model, pgon_dataloader, pgon_cla_dataloader, task, task_loss_weight,
            optimizer, tb_writer, logger, model_file, grt_epoches, cla_epoches, log_every, val_every, tol = 1e-6, 
            do_polygon_random_start = True, do_data_augment = False, global_batch_idx = 0, save_eval = False, pgon_flag = "simple"):
    '''
    Args:
        model: model.pgon_enc_dec is the one we will train
        pgon_dataloader: DataLoader() for polygon generator
        pgon_cla_dataloader: dict()
            key: ["TRAIN", "VALID", "TEST"]
            value: DataLoader() for polygon classification
        task: list(), ["grt", 'cla']
    '''
    ema_loss = None
    ema_losses = []
    losses = []

    running_loss = 0.0

    train_acc_list = []
    val_acc_list = []

    
    if "grt" in task:
        assert pgon_dataloader is not None
        for epoch in range(grt_epoches):
            
            for batch_idx, batch_data in enumerate(pgon_dataloader):
                if pgon_flag == "simple":
                    '''
                    pgon_ids: tensor.LongTensor(), shape (batch_size)
                    polygons: tensor.FloatTensor(), shape (batch_size, num_vert + 1, coord_dim = 2)
                    '''
                    pgon_ids, polygons = batch_data
                    global_batch_idx = epoch * len(pgon_dataloader) + batch_idx

                    if do_data_augment:
                        polygons = polygon_data_augment(polygons, data_augment_type = args.data_augment_type, device = args.device)
                        # polygons = random_flip_rotate_scale_polygons(polygons, device = args.device)

                    model.pgon_enc_dec.train()
                    optimizer.zero_grad()
                    loss = model.pgon_enc_dec.generative_loss(polygons, do_polygon_random_start = do_polygon_random_start)
                    
                    losses, ema_loss = update_loss(loss.item(), losses, ema_loss)
                    ema_losses.append(ema_loss)
                    running_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                    if global_batch_idx % log_every == 0:    # every 1000 mini-batches...
                        if tb_writer is not None:
                            tb_writer.add_scalar('ema_loss',
                                            ema_loss,
                                            global_batch_idx)
                            tb_writer.add_scalar('running_loss',
                                            running_loss/log_every,
                                            global_batch_idx)
                        logger.info("Epoch: {:d}; Iter: {:d}; Train generative running_loss: {:f}; ema_loss: {:f}".format(epoch, global_batch_idx, running_loss, ema_loss))

                        # # ...log a Matplotlib Figure showing the model's predictions on a
                        # # random mini-batch
                        # writer.add_figure('predictions vs. actuals',
                        #                 plot_classes_preds(net, inputs, labels),
                        #                 global_step=epoch * len(trainloader) + i)
                        running_loss = 0.0


                    if global_batch_idx % val_every == 0:
                        if model_file is not None:
                            save_model(model, optimizer, global_batch_idx, args, model_file)

                    # if check_conv(ema_losses, window = 5, tol=tol):
                    #     logger.info("Fully converged at Epoch: {:d}, iteration {:d}".format(epoch, global_batch_idx))
                    #     return
    else:
        grt_epoches = 0

    # if do_data_augment:
    #     theta = np.random.uniform(0, 2*np.pi)
    if "cla" in task:
        assert pgon_cla_dataloader is not None
        for epoch in range(cla_epoches):
            
            for batch_idx, batch_data in enumerate(pgon_cla_dataloader["TRAIN"]):
                if pgon_flag == "simple":
                    '''
                    pgon_ids: tensor.LongTensor(), shape (batch_size)
                    polygons: tensor.FloatTensor(), shape (batch_size, num_vert + 1, coord_dim = 2)
                    labels: tensor.LongTensor(), shape (batch_size), ground truth label
                    '''
                    pgon_ids, polygons, labels = batch_data
                    global_batch_idx += 1

                    if do_data_augment:
                        old_batch_size, _, _ = polygons.shape
                        polygons = polygon_data_augment(polygons, data_augment_type = args.data_augment_type, device = args.device)
                        new_batch_size, _, _ = polygons.shape
                        num_repeat = int(new_batch_size/old_batch_size)
                        labels = torch.repeat_interleave(labels.unsqueeze(0), repeats = num_repeat, dim=0).reshape(-1)
                        
                    #     if global_batch_idx > 1500:
                    #         polygons = random_flip_rotate_scale_polygons(polygons, theta = theta, device = args.device)
                    #         if global_batch_idx % 200 == 0:
                    #             # update theta every 200 iterator
                    #             theta = np.random.uniform(0, 2*np.pi)

                    
                    model.pgon_classifer.train()
                    optimizer.zero_grad()

                    if "grt" in task and "cla" in task:
                        cla_loss = model.pgon_classifer.forward(polygons, labels, do_polygon_random_start = do_polygon_random_start)
                        grt_loss = model.pgon_enc_dec.generative_loss(polygons, do_polygon_random_start = do_polygon_random_start)
                        loss = (1 - task_loss_weight) * grt_loss + task_loss_weight * cla_loss
                    elif task == ["cla"]:
                        loss = model.pgon_classifer.forward(polygons, labels, do_polygon_random_start = do_polygon_random_start)
                elif pgon_flag == "complex":
                    '''
                    pgon_ids: torch.LongTensor(), shape (batch_size)
                    V: torch.FloatTensor, shape (batch_size, num_vert, 2). vertex coordinates
                    E: torch.LongTensor, shape (batch_size, num_vert, 2). vertex connection, edge
                    labels: torch.LongTensor(), shape (batch_size), ground truth label
                    '''
                    pgon_ids, V, E, labels = batch_data
                    global_batch_idx += 1

                    if do_data_augment:
                        old_batch_size, _, _ = V.shape
                        # V: torch.FloatTensor, shape [batch_size * (1+num_aug), num_vert, 2]. vertex coordinates
                        V = polygon_data_augment(polygons = V, data_augment_type = args.data_augment_type, device = args.device)
                        new_batch_size, _, _ = V.shape
                        num_repeat = int(new_batch_size/old_batch_size)
                        # labels: torch.LongTensor(), shape [batch_size  * (1+num_aug)], ground truth label
                        labels = torch.repeat_interleave(labels.unsqueeze(0), repeats = num_repeat, dim=0).reshape(-1)
                        _, num_vert, ndim = E.shape
                        # E: torch.LongTensor, shape [batch_size * (1+num_aug), num_vert, 2]. vertex connection, edge
                        E = torch.repeat_interleave(E.unsqueeze(0), repeats = num_repeat, dim=0).reshape(-1,num_vert, ndim)
                        

                    
                    model.pgon_classifer.train()
                    optimizer.zero_grad()

                    if "grt" in task and "cla" in task:
                        raise Exception("We have not implement generater for complex polygon")
                        # cla_loss = model.pgon_classifer.forward(polygons, labels, do_polygon_random_start = do_polygon_random_start)
                        # grt_loss = model.pgon_enc_dec.generative_loss(polygons, do_polygon_random_start = do_polygon_random_start)
                        # loss = (1 - task_loss_weight) * grt_loss + task_loss_weight * cla_loss
                    elif task == ["cla"]:
                        loss = model.pgon_classifer.forward(polygons = None, 
                                                            labels = labels, 
                                                            do_polygon_random_start = do_polygon_random_start, 
                                                            V = V, E = E)
                
                losses, ema_loss = update_loss(loss.item(), losses, ema_loss)
                ema_losses.append(ema_loss)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()

                if global_batch_idx % log_every == 0:    # every 1000 mini-batches...
                    if tb_writer is not None:
                        tb_writer.add_scalar('ema_loss',
                                        ema_loss,
                                        global_batch_idx)
                        tb_writer.add_scalar('running_loss',
                                        running_loss/log_every,
                                        global_batch_idx)
                    if "grt" in task and "cla" in task:
                        logger.info("Epoch: {:d}; Iter: {:d}; Train generative_loss: {:f}; classification loss: {:f}; running_loss: {:f}; ema_loss: {:f}".format(
                                        epoch + grt_epoches, global_batch_idx, grt_loss.item(), cla_loss.item(), running_loss, ema_loss))
                    elif task == ["cla"]:
                        logger.info("Epoch: {:d}; Iter: {:d}; Train classification running_loss: {:f}; ema_loss: {:f}".format(
                                        epoch + grt_epoches, global_batch_idx, running_loss, ema_loss))

                    
                    running_loss = 0.0


                if global_batch_idx % val_every == 0:
                    train_preds, train_labels, train_pgon_ids, train_acc = eval_polygon_cla(model, pgon_cla_dataloader = pgon_cla_dataloader["TRAIN"], pgon_flag = pgon_flag)
                    if tb_writer is not None:
                        tb_writer.add_scalar('TRAIN_Accuracy',
                                        train_acc,
                                        global_batch_idx)
                    logger.info("Epoch: {:d}; Iter: {:d}; TRAIN accuracy: {:f}".format(
                                    epoch + grt_epoches, global_batch_idx, train_acc))

                    

                    
                    val_preds, val_labels, val_pgon_ids, val_acc = eval_polygon_cla(model, pgon_cla_dataloader = pgon_cla_dataloader["VALID"], pgon_flag = pgon_flag)
                    if tb_writer is not None:
                        tb_writer.add_scalar('VALID_Accuracy',
                                        val_acc,
                                        global_batch_idx)
                    logger.info("Epoch: {:d}; Iter: {:d}; VALID accuracy: {:f}".format(
                                    epoch + grt_epoches, global_batch_idx, val_acc))
                    

                    if model_file is not None:
                        if len(val_acc_list) == 0:
                            save_model(model, optimizer, global_batch_idx, args, model_file)
                        elif val_acc > np.max(val_acc_list):
                            save_model(model, optimizer, global_batch_idx, args, model_file)

                    train_acc_list.append(train_acc)
                    val_acc_list.append(val_acc)

        # model, optimizer, args = load_model(model, optimizer, model_file)
        # logger.info("The Best Model Evaluation:")
        # train_preds, train_labels, train_pgon_ids, train_acc = eval_polygon_cla(model, pgon_cla_dataloader = pgon_cla_dataloader["TRAIN"])          
        # logger.info("Epoch: {:d}; Iter: {:d}; TRAIN accuracy: {:f}".format(
        #                             epoch + grt_epoches, global_batch_idx, train_acc))

        # test_preds, test_labels, test_pgon_ids, test_acc = eval_polygon_cla(model, pgon_cla_dataloader = pgon_cla_dataloader["TEST"])
        # logger.info("Epoch: {:d}; Iter: {:d}; TEST accuracy: {:f}".format(
        #                                 epoch + grt_epoches, global_batch_idx, test_acc))

        eval_dict = eval_pgon_cla_model(args, model, optimizer, model_file, pgon_cla_dataloader, 
                            logger = logger,
                            epoch = epoch, 
                            grt_epoches = grt_epoches, 
                            global_batch_idx = global_batch_idx, 
                            save_eval = save_eval,
                            pgon_flag = pgon_flag)
        logger.info(f"Model save in {model_file}")

    return global_batch_idx, eval_dict

def eval_pgon_cla_model(args, model, optimizer, model_file, pgon_cla_dataloader, logger, 
            epoch = 0, grt_epoches = 0, global_batch_idx = 0, save_eval = False, pgon_flag = "simple"):
    model, optimizer, args = load_model(model, optimizer, model_file)
    logger.info("The Best Model Evaluation:")
    train_preds, train_labels, train_ids, train_acc = eval_polygon_cla(model, pgon_cla_dataloader = pgon_cla_dataloader["TRAIN"], pgon_flag = pgon_flag)          
    logger.info("Epoch: {:d}; Iter: {:d}; TRAIN accuracy: {:f}".format(
                                epoch + grt_epoches, global_batch_idx, train_acc))

    val_preds, val_labels, val_ids, val_acc = eval_polygon_cla(model, pgon_cla_dataloader = pgon_cla_dataloader["VALID"], pgon_flag = pgon_flag)          
    logger.info("Epoch: {:d}; Iter: {:d}; VALID accuracy: {:f}".format(
                                epoch + grt_epoches, global_batch_idx, val_acc))

    test_preds, test_labels, test_ids, test_acc = eval_polygon_cla(model, pgon_cla_dataloader = pgon_cla_dataloader["TEST"], pgon_flag = pgon_flag)
    logger.info("Epoch: {:d}; Iter: {:d}; TEST accuracy: {:f}".format(
                                    epoch + grt_epoches, global_batch_idx, test_acc))

    eval_dict = {"train": [train_preds, train_labels, train_ids, train_acc],
                    "val": [val_preds, val_labels, val_ids, val_acc],
                    "test": [test_preds, test_labels, test_ids, test_acc]}
    if save_eval:
        pickle_dump(eval_dict, model_file.replace(".pth", "__eval.pkl"))
         
    return eval_dict


def eval_polygon_cla(model, pgon_cla_dataloader, pgon_flag):
    '''
    pgon_cla_dataloader: the VALID or TEST dataloader for polygon classification
    '''
    pgon_ids_list = []
    preds = []
    label_list = []
    for batch_idx, batch_data in enumerate(pgon_cla_dataloader):
        if pgon_flag == "simple":
            '''
            pgon_ids: tensor.LongTensor(), shape (batch_size)
            polygons: tensor.FloatTensor(), shape (batch_size, num_vert + 1, coord_dim = 2)
            labels: tensor.LongTensor(), shape (batch_size), ground truth label
            '''
            pgon_ids, polygons, labels = batch_data
            with torch.no_grad():
                model.pgon_classifer.eval()

                # class_pred: the prediction for each class, shape (batch_size, num_classes)
                class_pred, pgon_embeds = model.pgon_classifer.compute_class_pred(polygons, do_polygon_random_start = False)

                
                class_pred = class_pred.detach().cpu().numpy()

                # pred: the predicted class labels, shape (batch_size)
                pred = np.argmax(class_pred, axis = 1)
                preds.append(pred)

                labels = labels.detach().cpu().numpy()
                label_list.append(labels)

                pgon_ids_list.append(pgon_ids.detach().cpu().numpy())

        elif pgon_flag == "complex":
            '''
            pgon_ids: torch.LongTensor(), shape (batch_size)
            V: torch.FloatTensor, shape (batch_size, num_vert, 2). vertex coordinates
            E: torch.LongTensor, shape (batch_size, num_vert, 2). vertex connection, edge
            labels: torch.LongTensor(), shape (batch_size), ground truth label
            '''
            pgon_ids, V, E, labels = batch_data
            with torch.no_grad():
                model.pgon_classifer.eval()

                # class_pred: the prediction for each class, shape (batch_size, num_classes)
                class_pred, pgon_embeds = model.pgon_classifer.compute_class_pred(polygons = None, do_polygon_random_start = False, 
                                                            V = V, E = E)

                
                class_pred = class_pred.detach().cpu().numpy()

                # pred: the predicted class labels, shape (batch_size)
                pred = np.argmax(class_pred, axis = 1)
                preds.append(pred)

                labels = labels.detach().cpu().numpy()
                label_list.append(labels)

                pgon_ids_list.append(pgon_ids.detach().cpu().numpy())

    
    # preds: the model prediction for all data sample in pgon_cla_dataloader
    preds = np.concatenate(preds)
    # label_list: the ground truth label for all data sample in pgon_cla_dataloader
    label_list = np.concatenate(label_list)
    # pgon_ids_list: their polygon ID list
    pgon_ids_list = np.concatenate(pgon_ids_list)

    acc = accuracy_score(y_true = label_list, y_pred = preds)
    
    return preds, label_list, pgon_ids_list, acc




def train_image_model(args, model, img_cla_dataloader, 
            optimizer, tb_writer, logger, model_file, cla_epoches, log_every, val_every, tol = 1e-6, 
            global_batch_idx = 0):

    loss_function = nn.CrossEntropyLoss()

    ema_loss = None
    ema_losses = []
    losses = []

    running_loss = 0.0

    train_acc_list = []
    val_acc_list = []
    
    assert img_cla_dataloader is not None
    for epoch in range(cla_epoches):
        
        for batch_idx, (img_ids, images, labels) in enumerate(img_cla_dataloader["TRAIN"]):
            '''
            img_ids: tensor.LongTensor(), shape (batch_size)
            images: tensor.FloatTensor(), shape (batch_size, num_channels, height, width)
            labels: tensor.LongTensor(), shape (batch_size), ground truth label
            '''
            global_batch_idx += 1

            # if do_data_augment:
            #     if global_batch_idx > 1500:
            #         polygons = random_flip_rotate_scale_polygons(polygons, theta = theta, device = args.device)
            #         if global_batch_idx % 200 == 0:
            #             # update theta every 200 iterator
            #             theta = np.random.uniform(0, 2*np.pi)


            model.train()
            optimizer.zero_grad()

            # logits: shape (batch_size, num_classes)
            logits = model.forward(images)
            loss = loss_function(logits, labels)
            
            losses, ema_loss = update_loss(loss.item(), losses, ema_loss)
            ema_losses.append(ema_loss)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            if batch_idx % log_every == 0:    # every 1000 mini-batches...
                if tb_writer is not None:
                    tb_writer.add_scalar('ema_loss',
                                    ema_loss,
                                    global_batch_idx)
                    tb_writer.add_scalar('running_loss',
                                    running_loss/log_every,
                                    global_batch_idx)
                
                
                logger.info("Epoch: {:d}; Iter: {:d}; Train classification running_loss: {:f}; ema_loss: {:f}".format(
                                    epoch, global_batch_idx, running_loss/log_every, ema_loss))

                
                running_loss = 0.0


            if batch_idx % val_every == 0:
                train_preds, train_labels, train_ids, train_acc = eval_image_cla(model, img_cla_dataloader = img_cla_dataloader["TRAIN"])
                if tb_writer is not None:
                    tb_writer.add_scalar('TRAIN_Accuracy',
                                    train_acc,
                                    global_batch_idx)
                logger.info("Epoch: {:d}; Iter: {:d}; TRAIN accuracy: {:f}".format(
                                epoch, global_batch_idx, train_acc))

                

                
                val_preds, val_labels, val_ids, val_acc = eval_image_cla(model, img_cla_dataloader = img_cla_dataloader["VALID"])
                if tb_writer is not None:
                    tb_writer.add_scalar('VALID_Accuracy',
                                    val_acc,
                                    global_batch_idx)
                logger.info("Epoch: {:d}; Iter: {:d}; VALID accuracy: {:f}".format(
                                epoch, global_batch_idx, val_acc))
                

                if model_file is not None:
                    if len(val_acc_list) == 0:
                        save_model(model, optimizer, global_batch_idx, args, model_file)
                    elif val_acc > np.max(val_acc_list):
                        save_model(model, optimizer, global_batch_idx, args, model_file)

                train_acc_list.append(train_acc)
                val_acc_list.append(val_acc)

    model, optimizer, args = load_model(model, optimizer, model_file)
    logger.info("The Best Model Evaluation:")
    train_preds, train_labels, train_ids, train_acc = eval_image_cla(model, img_cla_dataloader = img_cla_dataloader["TRAIN"])
    logger.info("Epoch: {:d}; Iter: {:d}; TRAIN accuracy: {:f}".format(
                                epoch, global_batch_idx, train_acc))


    test_preds, test_labels, test_ids, test_acc = eval_image_cla(model, img_cla_dataloader = img_cla_dataloader["TEST"])
    logger.info("Epoch: {:d}; Iter: {:d}; TEST accuracy: {:f}".format(
                                    epoch, global_batch_idx, test_acc))            
  

    return global_batch_idx



def eval_image_cla(model, img_cla_dataloader):
    '''
    img_cla_dataloader: the VALID or TEST dataloader for polygon classification
    '''
    img_ids_list = []
    preds = []
    label_list = []
    for batch_idx, (img_ids, images, labels) in enumerate(img_cla_dataloader):
        '''
        img_ids: tensor.LongTensor(), shape (batch_size)
        images: tensor.FloatTensor(), shape (batch_size, num_channels, height, width)
        labels: tensor.LongTensor(), shape (batch_size), ground truth label
        '''
        with torch.no_grad():
            model.eval()

            # class_pred: the prediction for each class, shape (batch_size, num_classes)
            class_pred = model.forward(images)

            
            class_pred = class_pred.detach().cpu().numpy()

            # pred: the predicted class labels, shape (batch_size)
            pred = np.argmax(class_pred, axis = 1)
            preds.append(pred)

            labels = labels.detach().cpu().numpy()
            label_list.append(labels)

            img_ids_list.append(img_ids.detach().cpu().numpy())



    
    # preds: the model prediction for all data sample in img_cla_dataloader
    preds = np.concatenate(preds)
    # label_list: the ground truth label for all data sample in img_cla_dataloader
    label_list = np.concatenate(label_list)
    # img_ids_list: their polygon ID list
    img_ids_list = np.concatenate(img_ids_list)

    acc = accuracy_score(y_true = label_list, y_pred = preds)
    
    return preds, label_list, img_ids_list, acc




def train_rel_cla_model(args, model, triple_dataloader, task, 
            optimizer, tb_writer, logger, model_file, cla_epoches, log_every, val_every, tol = 1e-6, 
            global_batch_idx = 0, save_eval = False, pgon_flag = "simple"):

    # loss_function = nn.CrossEntropyLoss()

    ema_loss = None
    ema_losses = []
    losses = []

    running_loss = 0.0

    train_acc_list = []
    val_acc_list = []
    
    assert triple_dataloader is not None
    for epoch in range(cla_epoches):
        
        for batch_idx, batch_data in enumerate(triple_dataloader["TRAIN"]):
            if pgon_flag == "simple":
                '''
                triple_ids: tensor, [batch_size]
                sids, rids, oids: tensor, [batch_size]
                sub_pgons: tensor, shape [batch_size, num_vert+1, 2]
                obj_pgons: tensor, shape [batch_size, num_vert+1, 2]
                labels: tensor, [batch_size], the target label, 
                '''
                triple_ids, sids, rids, oids, sub_pgons, obj_pgons, labels = batch_data
            elif pgon_flag == "complex":
                '''
                triple_ids: tensor, [batch_size]
                sids, rids, oids: tensor, [batch_size]
                sub_Vs, obj_Vs: float tensor, [batch_size, num_vert, 2]
                sub_Es, obj_Es: int tensor, [batch_size, num_vert, 2]
                labels: tensor, [batch_size], the target label, 
                '''
                triple_ids, sids, rids, oids, sub_Vs, sub_Es, obj_Vs, obj_Es, labels = batch_data
                
            global_batch_idx += 1

            # if do_data_augment:
            #     if global_batch_idx > 1500:
            #         polygons = random_flip_rotate_scale_polygons(polygons, theta = theta, device = args.device)
            #         if global_batch_idx % 200 == 0:
            #             # update theta every 200 iterator
            #             theta = np.random.uniform(0, 2*np.pi)


            model.train()
            optimizer.zero_grad()

            # logits: shape (batch_size, num_classes)
            if "rel" in task:
                if pgon_flag == "simple": 
                    loss = model.triple_dec.forward(sub_pgons, obj_pgons, labels, do_polygon_random_start = True)
                elif pgon_flag == "complex":
                    loss = model.triple_dec.forward(sub_pgons = None, obj_pgons = None, 
                                                labels = labels, 
                                                do_polygon_random_start = True,
                                                sub_Vs = sub_Vs, sub_Es = sub_Es, 
                                                obj_Vs = obj_Vs, obj_Es = obj_Es)
            losses, ema_loss = update_loss(loss.item(), losses, ema_loss)
            ema_losses.append(ema_loss)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            if global_batch_idx % log_every == 0:    # every 1000 mini-batches...
                if tb_writer is not None:
                    tb_writer.add_scalar('ema_loss',
                                    ema_loss,
                                    global_batch_idx)
                    tb_writer.add_scalar('running_loss',
                                    running_loss/log_every,
                                    global_batch_idx)
                
                
                logger.info("Epoch: {:d}; Iter: {:d}; Train classification running_loss: {:f}; ema_loss: {:f}".format(
                                    epoch, global_batch_idx, running_loss/log_every, ema_loss))

                
                running_loss = 0.0


            if global_batch_idx % val_every == 0:
                train_preds, train_labels, train_ids, train_acc = eval_rel_cla(model, 
                                                                    triple_dataloader = triple_dataloader["TRAIN"],
                                                                    pgon_flag = pgon_flag)
                if tb_writer is not None:
                    tb_writer.add_scalar('TRAIN_Accuracy',
                                    train_acc,
                                    global_batch_idx)
                logger.info("Epoch: {:d}; Iter: {:d}; TRAIN accuracy: {:f}".format(
                                epoch, global_batch_idx, train_acc))

                

                
                val_preds, val_labels, val_ids, val_acc = eval_rel_cla(model, 
                                                                    triple_dataloader = triple_dataloader["VALID"], 
                                                                    pgon_flag = pgon_flag)
                if tb_writer is not None:
                    tb_writer.add_scalar('VALID_Accuracy',
                                    val_acc,
                                    global_batch_idx)
                logger.info("Epoch: {:d}; Iter: {:d}; VALID accuracy: {:f}".format(
                                epoch, global_batch_idx, val_acc))
                

                if model_file is not None:
                    if len(val_acc_list) == 0:
                        save_model(model, optimizer, global_batch_idx, args, model_file)
                    elif val_acc > np.max(val_acc_list):
                        save_model(model, optimizer, global_batch_idx, args, model_file)

                train_acc_list.append(train_acc)
                val_acc_list.append(val_acc)

    eval_dict = eval_rel_cla_model(args, model, triple_dataloader, 
                        optimizer, logger, model_file, 
                        epoch = epoch, 
                        global_batch_idx = global_batch_idx, 
                        save_eval = save_eval,
                        pgon_flag = pgon_flag)

    return global_batch_idx, eval_dict


def eval_rel_cla_model(args, model, triple_dataloader, 
                    optimizer, logger, model_file, 
                    epoch = 0, global_batch_idx = 0, save_eval = False, pgon_flag = "simple"):


    model, optimizer, args = load_model(model, optimizer, model_file, device = args.device)
    logger.info("The Best Model Evaluation:")
    train_preds, train_labels, train_ids, train_acc = eval_rel_cla(model, triple_dataloader = triple_dataloader["TRAIN"], pgon_flag = pgon_flag)
    logger.info("Epoch: {:d}; Iter: {:d}; TRAIN accuracy: {:f}".format(
                                epoch, global_batch_idx, train_acc))

    val_preds, val_labels, val_ids, val_acc = eval_rel_cla(model, triple_dataloader = triple_dataloader["VALID"], pgon_flag = pgon_flag)
    logger.info("Epoch: {:d}; Iter: {:d}; VALID accuracy: {:f}".format(
                                epoch, global_batch_idx, val_acc))


    test_preds, test_labels, test_ids, test_acc = eval_rel_cla(model, triple_dataloader = triple_dataloader["TEST"], pgon_flag = pgon_flag)
    logger.info("Epoch: {:d}; Iter: {:d}; TEST accuracy: {:f}".format(
                                    epoch, global_batch_idx, test_acc))            
    
    eval_dict = {"train": [train_preds, train_labels, train_ids, train_acc],
                "val": [val_preds, val_labels, val_ids, val_acc],
                "test": [test_preds, test_labels, test_ids, test_acc]}
    if save_eval:
        pickle_dump(eval_dict, model_file.replace(".pth", "__eval.pkl"))

    return eval_dict



def eval_rel_cla(model, triple_dataloader, pgon_flag = "simple"):
    '''
    triple_dataloader: the VALID or TEST dataloader for polygon classification
    '''
    triple_ids_list = []
    preds = []
    label_list = []
    for batch_idx, batch_data in enumerate(triple_dataloader):
        if pgon_flag == "simple":
            '''
            triple_ids: tensor, [batch_size]
            sids, rids, oids: tensor, [batch_size]
            sub_pgons: tensor, shape [batch_size, num_vert+1, 2]
            obj_pgons: tensor, shape [batch_size, num_vert+1, 2]
            labels: tensor, [batch_size], the target label, 
            '''
            triple_ids, sids, rids, oids, sub_pgons, obj_pgons, labels = batch_data
        elif pgon_flag == "complex":
            '''
            triple_ids: tensor, [batch_size]
            sids, rids, oids: tensor, [batch_size]
            sub_Vs, obj_Vs: float tensor, [batch_size, num_vert, 2]
            sub_Es, obj_Es: int tensor, [batch_size, num_vert, 2]
            labels: tensor, [batch_size], the target label, 
            '''
            triple_ids, sids, rids, oids, sub_Vs, sub_Es, obj_Vs, obj_Es, labels = batch_data

        with torch.no_grad():
            model.eval()

            if pgon_flag == "simple": 
                # class_pred: the prediction for each class, shape (batch_size, num_classes)
                class_pred, sub_pgon_embeds, obj_pgon_embeds = model.triple_dec.compute_class_pred(
                                                sub_pgons, obj_pgons, 
                                                do_polygon_random_start = False, 
                                                do_softmax = False)
            elif pgon_flag == "complex":
                # class_pred: the prediction for each class, shape (batch_size, num_classes)
                class_pred, sub_pgon_embeds, obj_pgon_embeds = model.triple_dec.compute_class_pred(
                                                sub_pgons = None, obj_pgons = None, 
                                                do_polygon_random_start = False, 
                                                do_softmax = False,
                                                sub_Vs = sub_Vs, sub_Es = sub_Es, 
                                                obj_Vs = obj_Vs, obj_Es = obj_Es)
            
            class_pred = class_pred.detach().cpu().numpy()

            # pred: the predicted class labels, shape (batch_size)
            pred = np.argmax(class_pred, axis = 1)
            preds.append(pred)

            labels = labels.detach().cpu().numpy()
            label_list.append(labels)

            triple_ids_list.append(triple_ids.detach().cpu().numpy())



    
    # preds: the model prediction for all data sample in img_cla_dataloader
    preds = np.concatenate(preds)
    # label_list: the ground truth label for all data sample in img_cla_dataloader
    label_list = np.concatenate(label_list)
    # triple_ids_list: their polygon ID list
    triple_ids_list = np.concatenate(triple_ids_list)

    acc = accuracy_score(y_true = label_list, y_pred = preds)
    
    return preds, label_list, triple_ids_list, acc



