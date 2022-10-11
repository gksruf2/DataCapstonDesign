#!/bin/bash

#SBATCH --job-name my_capston
#SBATCH --gress=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --time 1-0
#SBATCH --mem-per-gpu=3G
#SBATCH --partition batch_ugrad
#SBATCH -x ai10
#SBATCH -o slurm/logs/slurm-%A-%x.out

python hello.py

exit 0
