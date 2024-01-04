import objaverse
import trimesh
import torch 
from torch.utils.data import Dataset
import numpy as np
uids = objaverse.load_uids()
annotations = objaverse.load_annotations(uids[:798759])
#annotations = objaverse.load_annotations(uids[:1000])
count = {}
for k, v in annotations.items():
    tags = v['tags']
    for tag in tags:
        name = tag['name']
        if name in count:
            count[name] += 1
        else:
            count[name] = 1

count = sorted(count.items(), key=lambda x:x[1], reverse=True)

for k, v in count:
    print("%s,%d"%(k, v))
