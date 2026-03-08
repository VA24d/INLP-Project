#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gnode[044-092]
#SBATCH --time=4-00:00:00
#SBATCH --output=muse_unlearn.out

source /home2/vijay.s/miniconda3/etc/profile.d/conda.sh
conda activate base

export HF_TOKEN="<REDACTED_HF_TOKEN>"
export CUDA_HOME=/usr/local/cuda

cd ~/inlp_project
python run_muse_ada.py
