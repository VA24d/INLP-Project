#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH --job-name=quant_base
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --exclude=gnode072,gnode091
#SBATCH -o /home2/vijay.s/slurm_logs/quant_base_%j.out
#SBATCH -e /home2/vijay.s/slurm_logs/quant_base_%j.err

set -euo pipefail
echo "=== quantize base model ==="
export HF_HOME=/home2/vijay.s/.cache/hf_inlp
export TRANSFORMERS_CACHE=/home2/vijay.s/.cache/hf_inlp
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

/home2/vijay.s/miniconda3/bin/python3 -u /home2/vijay.s/inlp_project/remote_sync/quantize_base.py
echo "=== done ==="
