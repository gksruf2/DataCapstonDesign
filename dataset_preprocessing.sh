#!/bin/bash

#SBATCH --job-name test
#SBATCH --time 1-0
#SBATCH --partition batch_sw_ugrad
#SBATCH -o logs/slurm-%A-%x.out
#SBATCH  --nodelist=sw9

python dataset_preprocessing.py

# letting slurm know this code finished without any problem

exit 0