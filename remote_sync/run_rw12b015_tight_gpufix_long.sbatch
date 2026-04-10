#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH --job-name=rw12b015_gplong
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:40:00
#SBATCH --exclude=gnode072,gnode091
#SBATCH -o /home2/vijay.s/inlp_project/remote_sync/run_rw12b015_tight_gpufix_long.out
#SBATCH -e /home2/vijay.s/inlp_project/remote_sync/run_rw12b015_tight_gpufix_long.err

set -euo pipefail
cd /home2/vijay.s/inlp_project

export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/home2/vijay.s/.cache/hf_inlp
export TRANSFORMERS_CACHE=/home2/vijay.s/.cache/hf_inlp
export HF_DATASETS_CACHE=/home2/vijay.s/.cache/hf_inlp/datasets
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

ROOT_DIR="/home2/vijay.s/inlp_project" \
PYTHON_BIN=/home2/vijay.s/miniconda3/bin/python3 \
MODEL_NAME=nightbloodredux/inlp-base-gemma3-1b-fp16 \
BASE_MODEL_PATH=nightbloodredux/inlp-base-gemma3-1b-fp16 \
ENH_EXPERIMENT=rw12b015_tight_gpufix_long \
ENH_USE_EXTERNAL_QA=1 \
ENH_EXTERNAL_QA_PATH=/home2/vijay.s/inlp_project/data/harrypotterqa \
ENH_EXTERNAL_QA_MAX=220 \
ENH_USE_EXTERNAL_TEXT_FORGET=0 \
ENH_TRAIN_FORGET_QA_MAX=260 \
ENH_TRAIN_RETAIN_QA_MAX=120 \
ENH_HARD_AUGMENT=1 \
ENH_HARD_AUG_VARIANTS=2 \
ENH_REFUSAL_CALIBRATION=1 \
ENH_REFUSAL_HARD_AUGMENT=1 \
ENH_REFUSAL_MAX_FORGET=96 \
ENH_REFUSAL_MAX_RETAIN=32 \
ENH_REFUSAL_EPOCHS=1 \
ENH_REFUSAL_TEXT="I can't help with Harry Potter specific details." \
ENH_DATALOADER_WORKERS=0 \
ENH_TRAINER_DATALOADER_WORKERS=0 \
ENH_MAP_NUM_PROC=0 \
ENH_TOKENIZE_INPUT_MAX_MULT=1 \
ENH_BATCH_SIZE=1 \
ENH_MAX_SEQ_LEN=96 \
ENH_REF_ON_CPU=0 \
ENH_TOTAL_PHASES=1 \
ENH_EPOCHS_PER_PHASE=1 \
ENH_MAX_STEPS=60 \
ENH_GRAD_ACCUM=4 \
ENH_BETA=0.22 \
ENH_RETAIN_WEIGHT=0.8 \
ENH_USE_SWA=0 \
ENH_USE_BS_NPO=0 \
ENH_USE_CNPO=1 \
ENH_CNPO_WEIGHT=0.28 \
ENH_CNPO_START_STEP=4 \
ENH_CNPO_INTERVAL=1 \
ENH_CNPO_TEMP=6.0 \
ENH_CNPO_RETAIN_PULL=0.5 \
ENH_USE_DATA_PARALLEL=0 \
ENH_SAVE_EVERY_STEPS=10 \
ENH_CHECKPOINT_DIR=/home2/vijay.s/inlp_project/models_enhanced/midcheckpoints_rw12b015_tight_gpufix_long \
MID_EVAL_STEP=30 \
RETRAIN_ENHANCED=1 \
SKIP_FULL_EVAL=1 \
DQA_ENABLE_ADVERSARIAL=1 \
DQA_ADV_MAX_ITEMS=24 \
ENH_REPORT_TO=none \
bash /home2/vijay.s/inlp_project/remote_sync/run_best_unlearn.sh
