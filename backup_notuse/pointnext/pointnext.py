import sys
import os
import argparse
pointnext_dir = './models/pointnext/PointNeXt'

def add_path_recursive(directory):
    sys.path.append(directory)
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            add_path_recursive(os.path.join(root, d))

add_path_recursive(pointnext_dir)

# print(sys.path)
from PointNeXt.openpoints.utils.config import EasyConfig
from PointNeXt.openpoints.models.build import build_model_from_cfg
from PointNeXt.openpoints.utils.ckpt_util import cal_model_parm_nums


def PointNEXT():

    cfg_path = '/home/hhfan/code/point-transformer/models/pointnext/PointNeXt/cfgs/objaverse/pointnext-s.yaml'

    cfg = EasyConfig()
    cfg.load(cfg_path, recursive=True)
    # print(f'cfg:{cfg}')

    model = build_model_from_cfg(cfg.model)
    model_size = cal_model_parm_nums(model)
    print("model size:")
    print(model_size)

    return model
