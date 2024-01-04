import os
import time
import csv
import json
import gzip
import objaverse
from collections import Counter
from skimage import io
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

num_points = 4096
glbs_folder = f'/mnt/data_sdb/obj/glbs' 
pcs_folder = f'/mnt/data_sdb/obj/pcs_{num_points}' 
metadata_foler = f'/mnt/data_sdb/obj/metadata'
json_path = '/mnt/data_sdb/obj/object-paths.json.gz'
uid_path_label_dict = {}



def load_annotations():
    # load uids and its annotations (mainly tags)
    uids = []
    for folder_id in os.listdir(pcs_folder):
        pcs_path = os.path.join(pcs_folder, folder_id) #/mnt/data_sdb/obj/glbs/000-000            
        for pc_file in os.listdir(pcs_path):  
            uid = os.path.splitext(pc_file)[0]
            uids.append(uid)
    annotations = objaverse.load_annotations(uids)
    # load uids and its paths
    with gzip.open(json_path, 'rt') as read_f:
        glb_path_dict = json.load(read_f)

            
    for uid in uids:
        path_label_dict = {}
        tags = annotations[uid]['tags'] # each object with a specific uid has multiple tags
        if len(tags)>0:
            label = tags[0]['name'] # choose the first tag's name as its label
            glb_path = glb_path_dict[uid]
            path = glb_path.replace('glbs', f'pcs_{num_points}')
            path = path.replace('.glb', '.npz')
            path_label_dict['label'] = label
            path_label_dict['path'] =  path
            uid_path_label_dict[uid] = path_label_dict       
        else:
            print(f'{uid} do not have tags')
        
        
    with gzip.open('./uid_path_label_dict.json.gz', "wt") as write_f:
        json.dump(uid_path_label_dict, write_f)
 
    
    
if __name__ == "__main__":
    load_annotations()