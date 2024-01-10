import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstraction

class PointNet2Loss(nn.Module):
    def __init__(self):
        super(PointNet2Loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
    
    
class PointNet2(nn.Module):
    def __init__(self,num_classes,normal_channel=True):
        super(PointNet2, self).__init__()
        in_channel = 7 if normal_channel else 3 # default 6
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm) # [b, 3, 512], [b, 128, 512]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # [b, 3, 512], [b, 256, 128]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # [b, 3, 1], [b, 1024, 1]

        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        # x: [b, num_classes], l3_points: [b, 1024, 1]
        return x, l3_points

if __name__ == '__main__':

    batch_size = 2
    num_points = 4096
    point_dim = 7
    input = torch.randn(batch_size, point_dim, num_points) # [b, d, n]
    
    model = PointNet2(
        num_classes = 908,
        normal_channel = True
    )
    x, l3_points = model(input)  
    
    print(f'x.shape:{x.shape}\nl3_points:{l3_points.shape}')


