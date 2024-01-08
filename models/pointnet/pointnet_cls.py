import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from .pointnet_utils import PointNetEncoder, feature_transform_reguliarzer

class PointNetLoss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(PointNetLoss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss

class PointNet(nn.Module):
    def __init__(self, num_classes=40, normal_channel=True):
        super(PointNet, self).__init__()
        channel = 7 if normal_channel else 3 # default 6
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        # output [b, num_classes], coordinate transformation matrix [b, 3, 3], feature transformation matrix [b, 64, 64]
        return x, trans_feat 
 



if __name__ == '__main__':

    model = PointNet(
        num_classes = 908,
        normal_channel = True
    )

    batch_size = 2
    num_points = 4096
    point_dim = 7

    input = torch.randn(batch_size, point_dim, num_points) # [b, d, n]

    x, trans_feat = model(input)  
    print(f'x.shape:{x.shape}\ntrans_feat:{trans_feat.shape}')