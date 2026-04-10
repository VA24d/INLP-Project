#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --constraint=2080ti
#SBATCH --time=2-00:00:00
#SBATCH --job-name=cnpo_ada_2080ti
#SBATCH --output=/home2/vijay.s/inlp_project/logs/cnpo_ada_%j.out
#SBATCH --error=/home2/vijay.s/inlp_project/logs/cnpo_ada_%j.err

set -euo pipefail

source /home2/vijay.s/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /home2/vijay.s/inlp_project
mkdir -p logs remote_sync data/harrypotterqa

echo "[INFO] Start time: $(date)"
echo "[INFO] Hostname: $(hostname)"

nvidia-smi -L || true
GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader | tr -d '\r')"
GPU_COUNT="$(echo "$GPU_NAMES" | sed '/^\s*$/d' | wc -l | xargs)"

echo "[INFO] GPU count detected: $GPU_COUNT"
echo "[INFO] GPU names:"
echo "$GPU_NAMES"

if [[ "$GPU_COUNT" -lt 4 ]]; then
  echo "[ERROR] Expected at least 4 GPUs, found $GPU_COUNT"
  exit 10
fi

if echo "$GPU_NAMES" | grep -vi "2080 Ti" >/dev/null; then
  echo "[ERROR] Non-2080Ti GPU detected. Aborting to keep run spec strict."
  exit 11
fi

if [[ ! -f data/harrypotterqa/harry_potter_qa.jsonl ]]; then
  echo "[ERROR] Missing QA file: data/harrypotterqa/harry_potter_qa.jsonl"
  exit 12
fi

echo "[INFO] QA lines: $(wc -l < data/harrypotterqa/harry_potter_qa.jsonl)"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY_BIN="$(command -v python3 || command -v python)"

ROOT_DIR="$PWD" \
PYTHON_BIN="$PY_BIN" \
MODEL_NAME=nightbloodredux/inlp-base-gemma3-1b-fp16 \
BASE_MODEL_PATH=nightbloodredux/inlp-base-gemma3-1b-fp16 \
ENH_EXPERIMENT=rw12b015_cnpo_qa270_ada \
ENH_USE_EXTERNAL_QA=1 \
ENH_EXTERNAL_QA_PATH="$PWD/data/harrypotterqa" \
ENH_EXTERNAL_QA_MAX=270 \
ENH_USE_EXTERNAL_TEXT_FORGET=1 \
ENH_EXTERNAL_TEXT_KAGGLE_DATASET=pratikshaaigal/harrypotterkb \
ENH_EXTERNAL_TEXT_MAX=240 \
ENH_EXTERNAL_TEXT_CLOZE_MAX=160 \
ENH_TRAIN_FORGET_QA_MAX=460 \
ENH_TRAIN_RETAIN_QA_MAX=320 \
ENH_DATALOADER_WORKERS=0 \
ENH_TRAINER_DATALOADER_WORKERS=0 \
ENH_MAP_NUM_PROC=0 \
ENH_TOKENIZE_INPUT_MAX_MULT=1 \
ENH_BATCH_SIZE=1 \
ENH_MAX_SEQ_LEN=96 \
ENH_REF_ON_CPU=1 \
ENH_REF_CPU_DTYPE=float16 \
ENH_TOTAL_PHASES=2 \
ENH_EPOCHS_PER_PHASE=2 \
ENH_MAX_STEPS=140 \
ENH_GRAD_ACCUM=4 \
ENH_BETA=0.22 \
ENH_RETAIN_WEIGHT=1.0 \
ENH_USE_SWA=0 \
ENH_USE_BS_NPO=0 \
ENH_USE_CNPO=1 \
ENH_CNPO_WEIGHT=0.22 \
ENH_CNPO_START_STEP=8 \
ENH_CNPO_INTERVAL=2 \
ENH_CNPO_TEMP=6.0 \
ENH_CNPO_RETAIN_PULL=0.6 \
ENH_HARD_AUGMENT=1 \
ENH_HARD_AUG_VARIANTS=3 \
ENH_REFUSAL_HARD_AUGMENT=1 \
ENH_REFUSAL_MAX_FORGET=128 \
ENH_REFUSAL_MAX_RETAIN=64 \
ENH_REFUSAL_EPOCHS=2 \
ENH_REPORT_TO=none \
DQA_ENABLE_ADVERSARIAL=1 \
DQA_ADV_MAX_ITEMS=24 \
bash remote_sync/run_best_unlearn.sh

echo "[INFO] End time: $(date)"