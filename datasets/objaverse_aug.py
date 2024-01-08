import os
import json
import objaverse
import trimesh
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import warnings
# warnings.filterwarnings('ignore')
import argparse

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(pc, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = pc.shape
    xyz = pc[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    pc = pc[centroids.astype(np.int32)]
    return pc


class ObjAugDataSet(Dataset):
    def __init__(self, pc_root, ann_path, args):
        self.pc_root = pc_root
        self.ann_path = ann_path

        self.split = args.split     
        self.process_data = args.process_data   
        self.npoints = args.num_points
        self.num_classes = args.num_classes
        self.uniform = args.use_uniform_sample # wether use FPS
        self.use_normals = args.use_normals # wether contain other dimensions

        self.data = self.load_data()
        print(f'The size of {self.split} data is {len(self.data)}')

        if self.uniform:
            self.save_path = os.path.join(pc_root, 'objaug%d_%s_%dpts_fps.dat' % (self.num_classes, self.split, self.npoints))
        else:
            self.save_path = os.path.join(pc_root, 'objaug%d_%s_%dpts.dat' % (self.num_classes, self.split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print(f'Processing data {self.save_path} (only running in the first time)...')
                self.list_of_points = [None] * len(self.data)
                self.list_of_labels = [None] * len(self.data)
                
                for index in tqdm(range(len(self.data)), total=len(self.data)):
                    pc = self.data[index]['pc']
                    label = self.data[index]['label']
                    
                    if self.uniform:
                        pc = farthest_point_sample(pc, self.npoints)
                    else:
                        pc = pc[0:self.npoints, :]
                        
                    self.list_of_points[index] = pc
                    self.list_of_labels[index] = label  
                 
                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def load_data(self):
        data = []
        with open(self.ann_path, 'r') as f:
            ann_data = json.load(f)
        for uid, info in ann_data.items():
            path = info['path']
            label = info['label']
            pc_path = os.path.join(self.pc_root, path)
            # print(f'pc_path:{pc_path}') 
            pc = np.load(pc_path)['points']
            data.append({
                'uid': uid,
                'pc': pc,
                'label': label
            })  
        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.process_data:
            pc, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            pc = self.data[index]['pc']
            label = self.data[index]['label']

            if self.uniform:
                pc = farthest_point_sample(pc, self.npoints)
            else:
                pc = pc[0:self.npoints, :]
                
        pc[:, 0:3] = pc_normalize(pc[:, 0:3])
        if not self.use_normals: # use_normals = True, pc:[b, n, 7]; use_normals = False, pc:[b, n, 3]
            pc = pc[:, 0:3]

        return pc, label

if __name__ == '__main__':
    pc_root = '/mnt/data_sdb/obj'
    ann_path = '/home/hhfan/code/pc/process/label/test_selected_4_uid_path_idx_dict.json'
    parser = argparse.ArgumentParser(description='PointTransformer Model training ...') 
    args = parser.parse_args()
    args.split = 'train'
    args.process_data = False 
    args.num_points = 1024   
    args.num_classes = 908
    args.use_normals = True # True dim = 7, False dim = 3
    args.use_uniform_sample = True
    data = ObjAugDataSet(pc_root=pc_root, ann_path=ann_path, args=args)
    DataLoader = DataLoader(data, batch_size=4, shuffle=True, num_workers=10)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
