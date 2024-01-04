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

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils.utils import *
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
    parser.add_argument('--output-dir', default='/home/hhfan/code/pc/results/', type=str, help='path where to save')
    parser.add_argument('--save-interval', default=40, type=int, help='save the checkpoint at (epoch+1) % n')
    # resume
    parser.add_argument('--resume', default='/mnt/data_sdb/pc_output/PointTransformer_119.pth', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')

    args = parser.parse_args()
    return args


def evaluate(model, criterion, accelerator, eval_dataloader, print_freq, eval_len):
    model.eval()
    
    metric_logger = MetricLogger(delimiter=" ")
    header = f'Eval:'
    outputs_list = []
    labels_list = []
  
    with torch.no_grad():    
        for uids, pcs, labels in metric_logger.log_every(eval_dataloader, print_freq, header):
            start_time = time.time()
            outputs = model(pcs)
            loss = criterion(outputs, labels)
            accelerator.backward(loss)
            
            outputs, labels = outputs.cpu().numpy().astype(np.int32), labels.cpu().numpy().astype(np.int32)
            outputs_list.append(outputs)
            labels_list.append(labels)
            
    accuracy = accuracy(labels_list, outputs_list)
    precision, recall, f1_score, _ = precision_recall_fscore_support(labels_list, outputs_list, average='weighted')
    return accuracy, precision, recall, f1_score    
    
            

def eval_model():
    args = parse_args()
    
    print('Creating data loader ...')
    eval_dataset = ObjDataset(pc_root=args.pc_root, ann_path=args.ann_path)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)  # pin_memory=True
    print('Creating PointTransformer model ...')
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
    
    if args.resume:
        print(f'from {args.resume} load the {args.model} ...')
        checkpoint = torch.load(args.resume, map_location='cpu')['model']
        model.load_state_dict(checkpoint)
    model.eval()
    print(f'finish construct {args.model}:\n{model}')        
               
    criterion = nn.CrossEntropyLoss()
  
    # accelerateor for multiple GPUs training
    accelerator = Accelerator()
    model, eval_dataloader,  = accelerator.prepare(
        model, eval_dataloader
    )
        
    print('Start evaluation ...')
    start_time = time.time()
    accuracy, precision, recall, f1_score  = evaluate(model, criterion, accelerator, eval_dataloader, args.print_freq, len(eval_dataset))
    end_time = time.time()
    
    metrics_list = [accuracy, precision, recall, f1_score]
    results = pd.DataFrame([metrics_list])
    mkdir(args.output_dir)
    results_path = args.output_dir + f'{args.model}_199_eval_1.csv'
    results.to_csv(results_path, mode='a', header=False, index=False)
    
    eval_time = end_time - start_time
    eval_time_str = str(datetime.timedelta(seconds=int(eval_time)))
    print(f'Total training time: {eval_time_str}')

if __name__ == '__main__':
    eval_model()
    