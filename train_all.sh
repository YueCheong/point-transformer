CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes=3 --main_process_port 29503 train_all.py --model PointNet2 -b 128 --lr 0.01 --num-points 4096 --use-uniform-sample True --output-dir /mnt/data_sdb/pv_output/log1 > logs/train_pointconv_1.log

# CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes=3 --main_process_port 29500 train_all.py --model PointNet -b 128 --lr 0.01 --output-dir /mnt/data_sdb/pn_output/log2 > logs/train_pointnet_2.log
# CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch train_all.py --model PointTransformer -b 64 --lr 0.01 --num-points 1024 --use-uniform-sample True --output-dir /mnt/data_sdb/pt_output/log2 >> logs/train_pointtransformer_2.log
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch train_all.py --model PointTransformer -b 16 --lr 0.01 --num-points 2048 --use-uniform-sample True --output-dir /mnt/data_sdb/pt_output/log3 >> logs/train_pointtransformer_3.log


CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes=3 --main_process_port 29500 train_all.py --model PointNet -b 128 --lr 0.01 --num-points 1024 --use-uniform-sample True --output-dir /mnt/data_sdb/pn_output/log3 >> logs/train_pointnet_3.log
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes=3 --main_process_port 29500 train_all.py --model PointNet -b 128 --lr 0.01 --num-points 2048 --use-uniform-sample True --output-dir /mnt/data_sdb/pn_output/log4 >> logs/train_pointnet_4.log

CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes=3 --main_process_port 29501 train_all.py --model PointNet2 -b 128 --lr 0.01 --num-points 1024 --use-uniform-sample True --output-dir /mnt/data_sdb/pn2_output/log2 >> logs/train_pointnet2_2.log
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes=3 --main_process_port 29501 train_all.py --model PointNet2 -b 128 --lr 0.01 --num-points 1024 --use-uniform-sample True --output-dir /mnt/data_sdb/pn2_output/log3 >> logs/train_pointnet2_3.log

CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes=3 --main_process_port 29502 train_all.py --model PointMLP -b 128 --lr 0.01 --use-xyz True --num-points 1024 --use-uniform-sample True --output-dir /mnt/data_sdb/pm_output/log2 >> logs/train_pointmlp_2.log
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes=3 --main_process_port 29502 train_all.py --model PointMLP -b 128 --lr 0.01 --use-xyz True --num-points 1024 --use-uniform-sample True --output-dir /mnt/data_sdb/pm_output/log3 >> logs/train_pointmlp_3.log


