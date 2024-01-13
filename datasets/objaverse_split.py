import os
import json
import objaverse
import trimesh
import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np


class ObjDataset(Dataset):
    def __init__(self, pc_root, ann_root, split_name):
        self.pc_root = pc_root
        self.ann_root = ann_root
        self.split_name = split_name
        self.data = self.load_data()
        
    def load_data(self):
        data = []     
        ann_path = os.path.join(self.ann_root, self.split_name)   
        with open(ann_path, 'r') as f:
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
        # uid = self.data[index]['uid']
        pc = self.data[index]['pc']
        label = self.data[index]['label'] 
        return pc.astype(np.float32), label   
        
      

if __name__ == '__main__':
    pc_root = '/mnt/data_sdb/obj'
    ann_root = '/home/hhfan/code/point-transformer/process/label/jsons'
    # ann_root = '/home/hhfan/code/point-transformer/process/label/uid_path_label_dict.json'
    split_name_list = []
    for split_id in range(10):
        split_name = f'ann_000-{split_id:03}.json'
        split_name_list.append(split_name)
        
    for split_name in split_name_list:
        print(f'Creating data loaders:{split_name}')    
        dataset = ObjDataset(pc_root=pc_root, ann_root=ann_root, split_name=split_name)
        data_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=10)  # pin_memory=True
    
        for pc, label in data_loader:
            # print(f'uid:{uid}')
            print(f'pc.shape:{pc.shape}') #[b, n, c]
            # print(f'label:{label}')

  



