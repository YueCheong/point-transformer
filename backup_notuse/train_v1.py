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
from datasets.objaverse import ObjDataset
from models.pointtransformer.point_transformer import PointTransformer

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='PointTransformer Model training ...') 
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--model', default='PointTransformer', type=str, help='model type')
    # dataset
    parser.add_argument('--pc-root', default='/mnt/data_sdb/obj', type=str, help='root path of point cloud files')  
    parser.add_argument('--ann-path', default='/home/hhfan/code/pc/process/label/test_selected_4_uid_path_idx_dict.json', type=str, help='annotation path') 
    # /home/hhfan/code/pc/process/label/uid_path_label_dict.json 
    # model
    parser.add_argument('--num_points', default=4096, type=int, help='number of points in each point cloud') 
    parser.add_argument('--num_classes', default=908, type=int, help='object categories of point clouds')  
    parser.add_argument('--dim', default=7, type=int, help='dimension of point cloud (transformer dim)')         
    parser.add_argument('--depth', default=12, type=int, help='depth of transformer layers')    
    parser.add_argument('--heads', default=8, type=int, help='head number in multi-head self-attention')      
    parser.add_argument('--mlp-dim', default=512, type=int, help='dimension MLP middle layer')      
    parser.add_argument('--pool', default='cls', type=str, help='pooling type of transformer (cls/mean)')  
    parser.add_argument('--dim-head', default=64, type=int, help='transformer dim for each head')      
    parser.add_argument('--dropout', default=0., type=float, help='dropout')
    parser.add_argument('--emb_dropout', default=0., type=float, help='dropout for embedding layer')                
    # training
    parser.add_argument('-b', '--batch-size', default=4, type=int, help='training batch size')
    parser.add_argument('--epochs', default=120, type=int, help='number of epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, help='number of data loading workers (default: 16)')    
    parser.add_argument('--lr',  default=0.05, type=float, help='initial learning rate (default: 0.05)')    
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for optimization, used for accelerated gradient descent')        
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)', dest='weight_decay')    
    parser.add_argument('--lr-milestones', nargs='+', default=[20, 35], type=int, help='decrease lr on milestones')        
    parser.add_argument('--lr-gamma', default=0.5, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=5, type=int, help='number of warmup epochs')        
    # output
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='/mnt/data_sdb/pc_output', type=str, help='path where to save')
    # resume
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')

    args = parser.parse_args()
    return args
     

def train_model():
    args = parse_args()
    accelerator = Accelerator()
    print("Creating data loaders ...")
    train_dataset = ObjDataset(pc_root=args.pc_root, ann_path=args.ann_path)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10)  # pin_memory=True
    print("Creating PointTransformer model ...")
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
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    warmup_iters = args.lr_warmup_epochs * len(train_dataloader)
    lr_milestones = [len(train_dataloader) * m for m in args.lr_milestones]
    scheduler = WarmupMultiStepLR(optimizer, milestones=lr_milestones, gamma=args.lr_gamma, warmup_iters=warmup_iters, warmup_factor=1e-5)
    
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    model.train()   
    
    start_time = datetime.now()
    for epoch in range(args.epochs):
        running_loss = 0.0
        
        for batch_idx, (uids, pcs, labels) in enumerate(train_dataloader):
            # print(f'batch_idx: {batch_idx}\nuids: {uids}\npcs: {pcs.shape}\nlabels: {labels})')
            # pcs, labels = pcs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(pcs)
            loss = criterion(outputs, labels)
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item() * pcs.size(0)
            print(f'Epoch [{epoch+1}/{args.epochs}] - Batch [{batch_idx+1}/{len(train_dataloader)}] - Loss: {loss.item():.4f}')
            
        epoch_loss = running_loss / len(train_dataset)
        print(f'Epoch [{epoch+1}/{args.epochs}] - Loss: {epoch_loss:.4f}')
    end_time = datetime.now()
    training_time = end_time - start_time
    print(f'Total training time: {training_time}')

if __name__ == '__main__':
    train_model()
    