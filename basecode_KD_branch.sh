#!/bin/bash

#SBATCH --job-name gksruf
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=5G
#SBATCH --time 3-0
#SBATCH --partition batch_sw_ugrad
#SBATCH -o logs/slurm-%A-%x.out
#SBATCH  --nodelist=sw9

#python FSRCNN_basecode_KD_branch.py --optimizer ADAM --epochs 100 --batch 2 --KDloss L1 --lr 0.0001
#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 200 --batch 32 --KDloss L2 --lr 0.0001 --t0 50 --name fsrcnn_b32_L2_t50_v1 #26~27

#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 200 --batch 4 --KDloss L1 --lr 0.0001 --t0 100 --name fsrcnn_b16_L1_t30_v1 # 28.4~28.4
#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 200 --batch 4 --KDloss L1 --lr 0.0001 --t0 100 --name fsrcnn_b16_L1_t30_v1 # 28.4~28.4

#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 200 --batch 16 --KDloss L2 --lr 0.0001 --t0 100 --name fsrcnn_b16_L2_t100_v2 #25~26
#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 100 --batch 32 --KDloss L2 --lr 0.0001 --t0 100 --name fsrcnn_b32_L2_t100_v1
#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 100 --batch 32 --KDloss L2 --lr 0.0001 --t0 100 --name fsrcnn_b32_L2_t100_v2
#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 100 --batch 16 --KDloss L2 --lr 0.0001 --t0 100 --name fsrcnn_b16_L2_t100_v1 #25

#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 200 --batch 16 --KDloss L2 --lr 0.0001 --t0 50 --name fsrcnn_b16_L2_t50_v2 #27~28
#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 100 --batch 32 --KDloss L2 --lr 0.0001 --t0 50 --name fsrcnn_b32_L2_t50_v2 #27
#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 100 --batch 16 --KDloss L2 --lr 0.0001 --t0 50 --name fsrcnn_b16_L2_t50_v1 #28
#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 200 --batch 32 --KDloss L1 --lr 0.0001 --t0 50 --name fsrcnn_b32_L1_t50_v2 #27.9~28.2
#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 200 --batch 32 --KDloss L1 --lr 0.0001 --t0 50 --name fsrcnn_b32_L1_t50_v2 #26.9~27.4
#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 200 --batch 16 --KDloss L1 --lr 0.0001 --t0 50 --name fsrcnn_b16_L1_t50_v1 #27.5~27.8
#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 200 --batch 16 --KDloss L1 --lr 0.0001 --t0 50 --name fsrcnn_b16_L1_t50_v1 #28.0~28.2

#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 200 --batch 32 --KDloss L2 --lr 0.0001 --t0 30 --name fsrcnn_b32_L2_t30_v2 # 26.7~26.6
#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 200 --batch 32 --KDloss L2 --lr 0.0001 --t0 30 --name fsrcnn_b32_L2_t30_v2 # 24.6~24.6
#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 200 --batch 16 --KDloss L2 --lr 0.0001 --t0 30 --name fsrcnn_b16_L2_t30_v1 # 27.6~27.7
#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 200 --batch 16 --KDloss L2 --lr 0.0001 --t0 30 --name fsrcnn_b16_L2_t30_v1 # 28.0~28.2
#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 200 --batch 32 --KDloss L1 --lr 0.0001 --t0 30 --name fsrcnn_b32_L1_t30_v2 # 27.4~27.4
#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 200 --batch 32 --KDloss L1 --lr 0.0001 --t0 30 --name fsrcnn_b32_L1_t30_v2 # 26.6~26.6
#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 200 --batch 16 --KDloss L1 --lr 0.0001 --t0 30 --name fsrcnn_b16_L1_t30_v1 # 28.4~28.4
#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 200 --batch 16 --KDloss L1 --lr 0.0001 --t0 30 --name fsrcnn_b16_L1_t30_v1 # 27.9~27.9

#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 200 --batch 8 --KDloss L1 --lr 0.0001 --t0 30 --name fsrcnn_b16_L1_t30_v1 # 28.4~28.4
#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 200 --batch 8 --KDloss L1 --lr 0.0001 --t0 30 --name fsrcnn_b16_L1_t30_v1 # 28.4~28.4
#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 200 --batch 8 --KDloss L1 --lr 0.0001 --t0 50 --name fsrcnn_b16_L1_t30_v1 # 28.4~28.4
#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 200 --batch 8 --KDloss L1 --lr 0.0001 --t0 50 --name fsrcnn_b16_L1_t30_v1 # 28.4~28.4
#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 200 --batch 8 --KDloss L1 --lr 0.0001 --t0 100 --name fsrcnn_b16_L1_t30_v1 # 28.4~28.4
#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 200 --batch 8 --KDloss L1 --lr 0.0001 --t0 100 --name fsrcnn_b16_L1_t30_v1 # 28.4~28.4

#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 200 --batch 4 --KDloss L1 --lr 0.0001 --t0 30 --name fsrcnn_b4_L1_t30_v1 # 28.4~28.4
#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 200 --batch 4 --KDloss L1 --lr 0.0001 --t0 30 --name fsrcnn_b4_L1_t30_v2 # 28.4~28.4
#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 200 --batch 4 --KDloss L2 --lr 0.0001 --t0 30 --name fsrcnn_b4_L2_t30_v1 # 28.4~28.4
#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 200 --batch 4 --KDloss L2 --lr 0.0001 --t0 30 --name fsrcnn_b4_L2_t30_v2 # 28.4~28.4
#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 200 --batch 2 --KDloss L1 --lr 0.0001 --t0 30 --name fsrcnn_b2_L1_t30_v1 # 28.4~28.4

# 64채널치 돌려봄..
#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 200 --batch 2 --KDloss L1 --lr 0.0001 --t0 30 --name fsrcnn_b2_L1_t30_64ver1 # 
#python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 200 --batch 2 --KDloss L1 --lr 0.0001 --t0 30 --name fsrcnn_b2_L1_t30_64ver1 # 

#python FSRCNN_basecode_KD_branch.py --optimizer ADAM --epochs 200 --batch 2 --KDloss L1 --lr 0.0005 --t0 30 --name fsrcnn_KD_b2_Lre-55_nt_v1 #

python FSRCNN_basecode_KD_branch.py --optimizer ADAM --epochs 500 --batch 2 --KDloss L1 --lr 0.0005 --t0 30 --name fsrcnn_KD_b2_Lre-55_nt500_nofe_v2 #

# letting slurm know this code finished without any problem

exit 0