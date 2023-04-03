#!/bin/bash

#SBATCH -p dgx1-gpu # partition
#SBATCH -J gen_dataset # job name
#SBATCH -o logs/gen_dataset.out # output/log file

#SBATCH -t 7-00:00:00 # d-hh:mm:ss
#SBATCH -n 4 # #cpu cores
#SBATCH --gres=gpu:0 # #gpus

name_prefix=gen_dataset
'bash' \
    ${name_prefix}.job \
    ../scripts/${name_prefix}.sh