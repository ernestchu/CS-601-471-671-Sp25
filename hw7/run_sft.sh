#!/bin/bash

#SBATCH --partition=mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:0
#SBATCH --job-name="CS 601.471/671 homework7"
#SBATCH --output=sft.out
#SBATCH --mem=16G

module load anaconda
conda activate hw7 # activate the Python environment

python3 run_sft.py --model HuggingFaceTB/SmolLM-360M \
    --dataset databricks/databricks-dolly-15k \
    --save_path $YOUR_SAVE_PATH \
    --batch_size 4 \
    --epochs 3 


    
