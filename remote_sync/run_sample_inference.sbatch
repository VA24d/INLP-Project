#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH --job-name=sample_infer
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclude=gnode072,gnode091
#SBATCH -o /home2/vijay.s/inlp_project/remote_sync/run_sample_inference.out
#SBATCH -e /home2/vijay.s/inlp_project/remote_sync/run_sample_inference.err

set -euo pipefail
cd /home2/vijay.s/inlp_project

export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/home2/vijay.s/.cache/hf_inlp
export TRANSFORMERS_CACHE=/home2/vijay.s/.cache/hf_inlp
export HF_HUB_DISABLE_XET=1

/home2/vijay.s/miniconda3/bin/python3 -u \
    /home2/vijay.s/inlp_project/remote_sync/sample_inference.py
