import os
import json
import objaverse
import trimesh
import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np


class ObjDataset(Dataset):
    def __init__(self, pc_root, ann_path):
        self.pc_root = pc_root
        self.ann_path = ann_path
        self.data = self.load_data()
        
    def load_data(self):
        data = []        
        with open(self.ann_path, 'r') as f:
            ann_data = json.load(f)            
        for uid, info in ann_data.items():
            path = info['path']
            label = info['label']   
            pc_path = os.path.join(self.pc_root, path)     
            print(f'pc_path:{pc_path}') 
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
        uid = self.data[index]['uid']
        pc = self.data[index]['pc']
        label = self.data[index]['label'] 
        return uid, pc.astype(np.float32), label   
        
      

if __name__ == '__main__':
    pc_root = '/mnt/data_sdb/obj'
    ann_path = '/home/hhfan/code/pc/process/label/uid_path_label_dict.json'
    dataset = ObjDataset(pc_root=pc_root, ann_path=ann_path)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=10)  # pin_memory=True
    
    for uid, pc, label in data_loader:
        print(f'uid:{uid}')
        print(f'pc.shape:{pc.shape}')
        # print(f'label:{label}')

  



