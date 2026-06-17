import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os.path as osp
import utils
from utils import AverageMeter
import mydataset
import argparse
import time
from model import get_model

import torch
import numpy as np
import myloss
from torch import nn
from torch.optim import Adam, SGD, lr_scheduler
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
import copy 
import matplotlib.pyplot as plt
from utils import saveImg,saveSingleImg
from sklearn.cluster import KMeans
import scipy.io as scio
from evaluate import evaluate as clustering_metric
# from constructGraph import getMvKNNGraph
from granular.base import MVGBList
from granular.granular_loss import MultiviewGCLoss


def train_1(loader, dataset, model,all_graph,all_encX, all_newX, loss_model, opt, sche, estimator, epoch,logger, result):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    rec_losses = AverageMeter()
    exi_losses = AverageMeter()
    all_ori = [torch.tensor(v_data) for v_data in dataset.mv_data]
    all_out = [torch.tensor([])]*dataset.view_num
    all_ind = torch.tensor([])
    all_com = torch.tensor([])

    all_label = dataset.cur_labels
    mse = nn.MSELoss()

    gra_loss = MultiviewGCLoss()

    model.train()
    if epoch < 1:
        model.recover = True
    else:
        model.recover = False
    end = time.time()
    for i, (data, label, inc_V_ind, _) in enumerate(loader):
        data_time.update(time.time() - end)
        data = [v_newX[i * args.batch_size:i * args.batch_size + label.size(0)].clone().to('cuda:0') for v_newX in all_newX]

        inc_V_ind=inc_V_ind.to('cuda:0')
        encX,decX,x_bar,H,emb_in,emb_out = model(copy.deepcopy(data), mask=inc_V_ind)
        hs = list(encX.unbind(dim=1))

        gen_weight = getattr(args, 'gen_weight', 0.1)
        confidence_mask = inc_V_ind * 1.0 + (1 - inc_V_ind) * gen_weight
        mv_gblist = MVGBList(hs, label, args.p, weight_mask=confidence_mask)

        con_loss = gra_loss(mv_gblist, weight_mask=confidence_mask)
        graph_loss = 0
        mse_loss = loss_model.weighted_wmse_loss(x_bar, data, inc_V_ind)

        loss_mse = mse_loss*args.beta
        loss_con = con_loss*args.alpha
        loss_graph = graph_loss*args.lambda1

        loss = loss_mse + loss_con
        opt.zero_grad()
        loss.backward()
        if isinstance(sche,CosineAnnealingWarmRestarts):
            sche.step(epoch + i / len(loader))
        
        opt.step()

        losses.update(loss.item())
        batch_time.update(time.time()- end)
        end = time.time()

        all_com = torch.cat((all_com, H.detach().clone().cpu()))
        all_out = [torch.cat((all_out[i],v_data.detach().clone().cpu()),0) for i,v_data in enumerate(x_bar)]
        all_ind = torch.cat((all_ind,inc_V_ind.detach().clone().cpu()),0)
        all_encX[i*args.batch_size:i*args.batch_size+label.size(0)] = encX.detach().clone()
    all_newX = copy.deepcopy(all_ori)
    for v,v_data in enumerate(all_out):
        all_newX[v][(1-all_ind[:,v]).bool()] = v_data[(1-all_ind[:,v]).bool()].clone().detach().cpu()

    acc, nmi, ari, pur, fscore = evaluate(all_com.numpy(), all_label, estimator, dataset.classes_num, epoch, logger)
    result["epoch"].append(epoch)
    result["loss_con"].append(loss_con)
    result["loss_rec"].append(loss_mse)
    result["ACC"].append(acc*100)
    result["NMI"].append(nmi*100)
    result["ARI"].append(ari*100)
    result["PUR"].append(pur*100)
    result["Fscore"].append(fscore*100)

    logger.info('Epoch:[{0}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss_mse {loss_mse:}\t'
                  'Loss_con {loss_con:}\t'
                  'exi_losses {exi_losses:}\t'
                  'Loss {losses.avg:}\t'.format(
                        epoch,   batch_time=batch_time,
                        data_time=data_time, loss_mse=loss_mse,
                        loss_con=loss_con, exi_losses=exi_losses.vals, losses=losses))
    return losses,model,result,all_encX,all_newX,all_label,all_ori,all_ind


def evaluate(H,all_label, estimator, classes_num, epoch,logger):
    end = time.time()
    preds = estimator.fit_predict(H)
    all_label = all_label.reshape(-1)
    acc, nmi, ari, pur, fscore = clustering_metric(all_label,preds)
    print('ACC:{:.2f}  NMI:{:.2f}  ARI:{:.2f}  PUR:{:.2f}  Fscore:{:.2f}'.format(acc*100, nmi*100, ari*100, pur*100, fscore*100))
    return acc, nmi, ari, pur, fscore


def main(args,file_path):
    data_path = osp.join(args.data_dir,args.dataset+'.mat')
    fold_data_path = osp.join(args.fold_dir, args.dataset+'_percentDel_'+str(args.mask_view_ratio)+'.mat')\
        if args.dataset !='animal' else osp.join(args.fold_dir, args.dataset+'_pairedrate_'+str(args.mask_view_ratio)+'.mat')

    folds_num = args.folds_num
    folds_results = [AverageMeter() for i in range(5)]
    if args.logs:
        logfile = osp.join(args.logs_dir,args.name+args.dataset+'_V_' + str(
                                    args.mask_view_ratio) + '_L_' +
                                    str(args.mask_label_ratio) + '_T_' + 
                                    str(args.training_sample_ratio) + '_'+str(args.beta)+'_'+str(args.gamma)+'.txt')
    else:
        logfile=None
    logger = utils.setLogger(logfile)
    
    for fold_idx in range(folds_num):
        train_dataloder,train_dataset = mydataset.getIncDataloader(data_path,fold_data_path,training_ratio=args.mask_view_ratio,fold_idx=fold_idx,is_train=True,batch_size=args.batch_size,shuffle = False,num_workers=args.workers)


        d_list = train_dataset.d_list
        classes_num = train_dataset.classes_num
        model = get_model(d_list,d_model=args.dim,n_layers=1,heads=4,classes_num=train_dataset.classes_num,dropout=0.)

        loss_model = myloss.MyLoss()

        optimizer = Adam(model.parameters(), lr=args.lr)

        scheduler = None

        estimator = KMeans(n_clusters=classes_num, max_iter=300, n_init=10, random_state=928)
        
        logger.info('train_data_num:'+str(len(train_dataset))+'   fold_idx:'+str(fold_idx))
        print(args)
        static_res = AverageMeter()
        epoch_results = [AverageMeter() for i in range(5)]
        total_losses = AverageMeter()
        train_losses_last = AverageMeter()
        all_newX = [torch.tensor(v_data,dtype=torch.float) for v_data in train_dataset.inc_mv_data]
        all_newX = [torch.tensor(v_data) for v_data in train_dataset.mv_data]
        all_encX = torch.ones((len(train_dataset),train_dataset.view_num,args.dim)).to('cuda:0')
        all_graph=None
        result = {
            "epoch": [],
            "loss_con": [],
            "loss_rec": [],
            "ACC": [],
            "NMI": [],
            "ARI": [],
            "PUR": [],
            "Fscore": []
        }
        for epoch in range(args.epochs):
            train_losses, model, result, all_encX, all_newX, all_label, all_ori, all_ind = train_1(train_dataloder, train_dataset, model, all_graph,
                                                              all_encX, all_newX,loss_model, optimizer, scheduler, estimator,
                                                              epoch, logger, result)
        best_epoch = np.argmax(result["ACC"])
        best_result = {
            "epoch": best_epoch,
            "ACC": result["ACC"][best_epoch],
            "NMI": result["NMI"][best_epoch],
            "ARI": result["ARI"][best_epoch],
            "PUR": result["PUR"][best_epoch],
            "Fscore": result["Fscore"][best_epoch],
            "loss_con": result["loss_con"][best_epoch],
            "loss_rec": result["loss_rec"][best_epoch],
        }
        print(f"best_result={best_result}")

def filterparam(file_path,index):
    params = []
    if os.path.exists(file_path):
        file_handle = open(file_path, mode='r')
        lines = file_handle.readlines()
        lines = lines[1:] if len(lines)>1 else []
        params = [[float(line.split(' ')[idx]) for idx in index] for line in lines ]
    return params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--logs-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--logs', default=False, type=bool)
    parser.add_argument('--records-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'records'))
    parser.add_argument('--file-path', type=str, metavar='PATH', 
                        default='')
    parser.add_argument('--data-dir', type=str, metavar='PATH', 
                        default='data/')
    parser.add_argument('--fold-dir', type=str, metavar='PATH', 
                        default='data/')
    parser.add_argument('--dataset', type=str, default='handwritten-5view')#handwritten-5view NH_jerry Caltech101-7 animal
    parser.add_argument('--mask_view_ratio', type=float, default=0.5)
    parser.add_argument('--folds-num', default=1, type=int) 
    parser.add_argument('--weights-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'weights'))
    parser.add_argument('--curve-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'output'))
    parser.add_argument('--img-dir', type=str, metavar='PATH', 
                        default='hw-imgs/0.5_')
    parser.add_argument('--save-curve', default=False, type=bool)
    parser.add_argument('--save-img', default=False, type=bool)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--workers', default=4, type=int)

    parser.add_argument('--name', type=str, default='final_')
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=101)
    parser.add_argument('--rec_epochs', type=int, default=50)
    
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--beta', type=float, default=1e-1)

    parser.add_argument('--p', type=float, default=2)
    parser.add_argument('--alpha', type=float, default=1e-1)
    parser.add_argument('--lambda1', type=float, default=1e-1)

    parser.add_argument('--gen-weight', type=float, default=0.1)

    
    args = parser.parse_args()
    file_path = osp.join(args.records_dir,args.name+str(args.epochs)+str(args.rec_epochs)+args.dataset+'_ViewMask_' + str(
                                    args.mask_view_ratio)+'.txt')
    args.file_path = file_path
    if args.logs:
        if not os.path.exists(args.logs_dir):
            os.makedirs(args.logs_dir)
    args.lr = 1e-3
    args.beta = 1
    args.alpha = 0.01
    main(args, file_path)

