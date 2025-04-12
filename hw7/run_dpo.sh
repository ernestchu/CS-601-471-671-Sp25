#!/bin/bash

#SBATCH --partition=mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --job-name="CS 601.471/671 homework7 DPO"
#SBATCH --output=dpo.out
#SBATCH --mem=16G

module load anaconda
conda activate hw7 # activate the Python environment

python run_dpo.py --model $YOUR_SAVE_PATH_OF_SFT_MODEL \
    --dataset dogtooth/helpsteer2_binarized_filtered \
    --epochs 2 \
    --save_path $YOUR_SAVE_PATH_OF_DPO_MODEL

