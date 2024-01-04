import objaverse
import trimesh
import torch 
from torch.utils.data import Dataset
import numpy as np


class ObjDataset(Dataset):
    def __init__(self, train=True):
        super(ObjDataset, self).__init__()

        self.uids = objaverse.load_uids()
        print("len(self.uids):", len(self.uids))
        self.annotations = objaverse.load_annotations(self.uids[:10])
        print("self.annotations[self.uids[0]]", self.annotations[self.uids[0]])
        self.cc_by_uids = [uid for uid, annotation in self.annotations.items() if annotation['license'] == 'by']
        print("self.cc_by_uids", self.cc_by_uids[:10])
        self.objects = objaverse.load_objects(self.cc_by_uids[:2]) # {uid, GLB_PATH}
        # self.objects = objaverse.load_objects([ , ]) # {uid, GLB_PATH}
        # self.lvis_annotations = objaverse.load_lvis_annotations()
        # print("lvis_annotations", self.lvis_annotations)

        self.train = train
        self.meshes = []
        self.points = []
        self.colors = []
        self.face_indexes = []
        self.labels = []
        
        for filename in self.objects:
            path = self.objects[filename]
            tag = self.annotations[filename]['tags']
            mesh = trimesh.load(path, force='mesh')
            point, face_index, color = trimesh.sample.sample_surface(mesh, count=4096, face_weight=None, sample_color=True, seed=0)
            self.point  = point
            self.face_index = face_index
            self.color = color
            self.points.append(np.array(point))
            self.colors.append(np.array(color))
            self.face_indexes.append(np.array(face_index))
            # self.labels.append()

        self.points = np.array(self.points)
        self.colors = np.array(self.colors)
        self.face_indexes = np.array(self.face_indexes)

        
        print("points.shape", self.points.shape)
        print("colors.shape", self.colors.shape)    
        print("face_indexes.shape", self.face_indexes.shape)

        
    def __len__(self):
        return len(self.points)
    
    def __getitem__(self, index):
        point = self.points[index]
        print("point in get_item ", point)
        color = self.colors[index]
        face_index = self.face_indexes[index]
        # label = self.label[index]
        return point.astype(np.float32), color.astype(np.float32), face_index.astype(np.int32) #, label.astype(np.int32)

if __name__ == '__main__':
    dataset = ObjDataset(train=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=10)  # pin_memory=True
    
    for points, colors, face_indexes in data_loader:
        print("points.shape", points.shape)
        print("colors.shape", colors.shape)
        print("face_indexes.shape", face_indexes.shape)
        # print("labels.shape", labels.shape)    



