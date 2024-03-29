import os
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.scheduler import WarmupMultiStepLR
import torch.distributed as dist
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from datasets.objaverse_split import ObjDataset
from datasets.objaverse_aug_split import ObjAugDataSet

from models.pointtransformer.point_transformer import PointTransformer
from models.pointnet.pointnet_cls import PointNet, PointNetLoss
from models.pointnet2.pointnet2_cls_ssg import PointNet2, PointNet2Loss
from models.pointmlp.pointmlp import pointMLP, pointMLPElite, cal_PointMLP_loss
from models.pointconv.pointconv_cls_ssg import PointConv, PointConvLoss
import numpy as np

from utils.utils import *
from utils.provider import *
import gc
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='PointTransformer Model training ...') 
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--model', default='PointTransformer', type=str, help='model type') # PointTransformer, PointNet, PointNet2
    # dataset
    parser.add_argument('--pc-root', default='/mnt/data_sdb/obj', type=str, help='root path of point cloud files')  
    parser.add_argument('--ann-root', default='/home/hhfan/code/point-transformer/process/label/jsons', type=str, help='root path of annotation files') 
    # /home/hhfan/code/pc/process/label/uid_path_label_dict.json 
    parser.add_argument('--process-data', default=False, type=bool, help='process and save data offline, or load the processed data')  
    parser.add_argument('--use-feature', default=True, type=bool, help='default True, dim > 3')
    parser.add_argument('--use-uniform-sample', default=False, type=bool, help='use uniform sampiling (FPS)')    
    parser.add_argument('--num-points', default=4096, type=int, help='number of points in each point cloud') 
    parser.add_argument('--num-classes', default=909, type=int, help='object categories of point clouds')  
    # pointtransformer model
    parser.add_argument('--dim', default=7, type=int, help='dimension of point cloud (transformer dim)')         
    parser.add_argument('--depth', default=12, type=int, help='depth of transformer layers')    
    parser.add_argument('--heads', default=8, type=int, help='head number in multi-head self-attention')      
    parser.add_argument('--mlp-dim', default=512, type=int, help='dimension MLP middle layer')      
    parser.add_argument('--pool', default='cls', type=str, help='pooling type of transformer (cls/mean)')  
    parser.add_argument('--dim-head', default=64, type=int, help='transformer dim for each head')      
    parser.add_argument('--dropout', default=0., type=float, help='dropout')
    parser.add_argument('--emb-dropout', default=0., type=float, help='dropout for embedding layer')                
    # pointnet, pointnet2 model
    
    # pointmlp args.use_xyz
    parser.add_argument('--use-xyz', default=False, type=bool, help='defaut not add xyz_dim (3)')
    
    # training
    parser.add_argument('--data-type', default='train', type=str, help='training / eval(test)')
    parser.add_argument('-b', '--batch-size', default=4, type=int, help='training batch size')
    parser.add_argument('--epochs', default=200, type=int, help='number of epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, help='number of data loading workers (default: 16)')    
    parser.add_argument('--optimizer',  default='SGD', type=str, help='optimizer')
    parser.add_argument('--lr',  default=0.05, type=float, help='initial learning rate (default: 0.05)')    
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for optimization, used for accelerated gradient descent')        
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)', dest='weight_decay')    
    parser.add_argument('--lr-milestones', nargs='+', default=[20, 35], type=int, help='decrease lr on milestones')        
    parser.add_argument('--lr-gamma', default=0.5, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=5, type=int, help='number of warmup epochs')        
    # output
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='/mnt/data_sdb/pc_output', type=str, help='path where to save')
    parser.add_argument('--save-interval', default=40, type=int, help='save the checkpoint at (epoch+1) % n')
    # resume
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')

    args = parser.parse_args()
    return args


def train_one_epoch(model, criterion, optimizer, scheduler, accelerator, epoch, split_name_list, args):
    model.train()
    
    metric_logger = MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('pcs/s', SmoothedValue(window_size=10, fmt='{value:.3f}'))
    header = f'Epoch: [{epoch}]'

    top1_acc_list = []    
    data_len = 0
    for split_name in split_name_list:
        print(f'Creating data loaders: {split_name}')
        train_dataset = ObjAugDataSet(pc_root=args.pc_root, ann_root=args.ann_root, split_name=split_name, args=args) # zy
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)  # pin_memory=True    
        data_len = data_len + len(train_dataloader)
        print(f'sample = len * batch_size =  {len(train_dataloader)} * {args.batch_size} = {len(train_dataloader)*args.batch_size}')        
                
        for pcs, labels in metric_logger.log_every(accelerator.prepare(train_dataloader), args.print_freq, header):
            start_time = time.time()
            optimizer.zero_grad()
            # data augmentation
            pcs = pcs.cpu().detach().numpy()
            pcs = random_point_dropout(pcs)
            pcs[:, :, 0:3] = random_scale_point_cloud(pcs[:, :, 0:3])
            pcs[:, :, 0:3] = shift_point_cloud(pcs[:, :, 0:3])
            pcs = torch.Tensor(pcs) # [b, n, d]
            
            if args.model == 'PointTransformer':           
                outputs = model(pcs)   
                loss = criterion(outputs, labels)
            elif args.model == 'PointNet':
                pcs = torch.transpose(pcs, 1, 2)
                outputs, trans_feat = model(pcs)
                loss = criterion(outputs, labels, trans_feat)
            elif args.model == 'PointNet2':
                pcs = torch.transpose(pcs, 1, 2) # [b, d, n]
                outputs, l3_points = model(pcs)
                loss = criterion(outputs, labels, l3_points) 
            elif args.model == 'PointMLP':
                pcs = torch.transpose(pcs, 1, 2) # [b, d, n]
                outputs = model(pcs)
                loss = criterion(outputs, labels)
            elif args.model == 'PointConv':
                pcs = torch.transpose(pcs, 1, 2)
                outputs = model(pcs[:, :3, :], pcs[:, 3:, :])  # xyz [b, 3, n], feat [b, 4, n]
                loss = criterion(outputs, labels)
            else:
                print(f'! Choose model type: PointTransformer, PointNet, PointNet2, PointMLP, PointConv')
                
                
            accelerator.backward(loss)
            optimizer.step()
        
            top1_acc = accuracy(outputs, labels, topk=(1,))
            top1_acc_list.append(top1_acc[0].item())
            
            batch_size = pcs.shape[0]
            metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
            metric_logger.meters['acc'].update(top1_acc[0].item(), n=batch_size)
            metric_logger.meters['pcs/s'].update(batch_size / (time.time() - start_time))
            
            scheduler.step()       
            sys.stdout.flush()
        
        # 释放资源
        torch.cuda.empty_cache()
        train_dataset = None
        del train_dataset
        # print(f'del dataset {split_name} and collect()')
        gc.collect() 
        
    print(f'total len of whole dataloader: {data_len}')  
    return top1_acc_list     
    
            
def train_model():
    args = parse_args()
    print(f'args:{args}')

    split_name_list = []
    for split_id in range(10):
        split_name = f'selected_4_idx_ann_000-{split_id:03}.json' # ZY 
        split_name_list.append(split_name)

    print('Creating model ...')
    if args.model == 'PointTransformer':
        model = PointTransformer(
            num_points = args.num_points,
            num_classes = args.num_classes,
            dim = args.dim,
            depth = args.depth,
            heads = args.heads,
            mlp_dim = args.mlp_dim,
            pool = args.pool,
            dim_head = args.dim_head, 
            dropout = args.dropout,
            emb_dropout = args.emb_dropout
        )
        criterion = nn.CrossEntropyLoss()
    elif args.model == 'PointNet':
        model = PointNet(
            num_classes = args.num_classes,
            normal_channel = args.use_feature
        )
        criterion = PointNetLoss()
    elif args.model == 'PointNet2':
        model = PointNet2(
            num_classes = args.num_classes,
            normal_channel = args.use_feature            
        )
        criterion = PointNet2Loss()
    elif args.model == 'PointMLP': # PointMLPElite
        model = pointMLP(   
            num_points = args.num_points,
            num_classes = args.num_classes,
            use_xyz = args.use_xyz 
        )
        criterion = cal_PointMLP_loss
    elif args.model == 'PointConv':
        model = PointConv(
            num_classes = args.num_classes,
            normal_channel = args.use_feature
        )
        criterion = PointConvLoss()
    else:
        print(f'! Choose model type: PointTransformer, PointNet, PointNet2, PointMLP, PointConv')
    
    if args.resume:
        print(f'Use pretrain model: {args.resume}')
        checkpoint = torch.load(args.resume, map_location='cpu')
        pre_state = checkpoint['model']
        update_dict = {k: v for k, v in pre_state.items() if k.startswith('module.tube_embedding.') or k.startswith('module.transformer1.') or k.startswith('module.pos')}
        for name in update_dict.keys():
            print(name)
        net_stat_dict = model.state_dict()
        net_stat_dict.update(update_dict)
        model.load_state_dict(net_stat_dict)
                

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=args.lr, 
            momentum=args.momentum, 
            weight_decay=args.weight_decay
        )
    else:
        print(f'! Choose optimizer type: Adam, SGD')

    # warmup_iters = args.lr_warmup_epochs * len(train_dataloader)
    # lr_milestones = [len(train_dataloader) * m for m in args.lr_milestones]
    warmup_iters = args.lr_warmup_epochs * 1000
    lr_milestones = [1000 * m for m in args.lr_milestones]
    scheduler = WarmupMultiStepLR(optimizer, milestones=lr_milestones, gamma=args.lr_gamma, warmup_iters=warmup_iters, warmup_factor=1e-5)
    
    # accelerateor for multiple GPUs training
    accelerator = Accelerator()
    model, optimizer, scheduler = accelerator.prepare(
        model, optimizer, scheduler
    )       
    print('Start training ...')
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        top1_acc_list = train_one_epoch(model, criterion, optimizer, scheduler, accelerator, epoch, split_name_list, args)
        if (epoch + 1) % args.save_interval == 0:
            if args.output_dir:
                mkdir(args.output_dir)
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args
                }
                save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, f'{args.model}_{epoch}.pth')
                )  
        epoch_top1_acc = np.mean((np.array(top1_acc_list)))   
        print(f'mean accuray of epoch {epoch} is {epoch_top1_acc}')   
        
    end_time = time.time()
    training_time = end_time - start_time
    training_time_str = str(datetime.timedelta(seconds=int(training_time)))
    print(f'Total training time: {training_time_str}')

if __name__ == '__main__':
    train_model()
    