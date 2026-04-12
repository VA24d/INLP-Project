#!/bin/bash
# =============================================================================
# simnpo_kl_v1 — SimNPO + Bias-NPO + KL-retain + Cosine-LR unlearning run
#
# Key fixes vs rw12b015_tight_gpufix:
#   - SimNPO (40% blend): length-normalised forget loss fixes "Marsreciate"
#     garbage-token appending by removing the token-count incentive.
#   - Bias-NPO (margin 0.3): extra forget push without more steps.
#   - KL-retain (weight 1.5): KL(ref||policy) on retain batches directly
#     anchors output distribution → prevents retain regression.
#   - Cosine-LR per phase: smoother optimisation trajectory.
#   - SWA enabled: smooths the noisy gradient-ascent landing point.
#   - 200 steps (5× gpufix) + 3 phases × 2 epochs: comprehensive HP forgetting.
#   - retain_weight raised 0.8 → 1.8: stronger GradDiff retain.
#   - CNPO disabled: avoid double ref_model forward on retain with KL active.
# =============================================================================
#SBATCH -A research
#SBATCH -p u22
#SBATCH --job-name=simnpo_kl_v1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:30:00
#SBATCH --exclude=gnode072,gnode091
#SBATCH -o /home2/vijay.s/inlp_project/remote_sync/run_simnpo_kl_v1.out
#SBATCH -e /home2/vijay.s/inlp_project/remote_sync/run_simnpo_kl_v1.err

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

export ROOT_DIR=/home2/vijay.s/inlp_project
export PYTHON_BIN=/home2/vijay.s/miniconda3/bin/python3
export MODEL_NAME=nightbloodredux/inlp-base-gemma3-1b-fp16
export BASE_MODEL_PATH=nightbloodredux/inlp-base-gemma3-1b-fp16
export ENH_EXPERIMENT=simnpo_kl_v1

export ENH_BATCH_SIZE=1
export ENH_MAX_SEQ_LEN=96
export ENH_GRAD_ACCUM=4
export ENH_DATALOADER_WORKERS=0
export ENH_TRAINER_DATALOADER_WORKERS=0
export ENH_MAP_NUM_PROC=0
export ENH_TOKENIZE_INPUT_MAX_MULT=1

export ENH_TOTAL_PHASES=3
export ENH_EPOCHS_PER_PHASE=2
export ENH_START_LAYERS=8
export ENH_UNFREEZE_PER_PHASE=4
export ENH_MAX_STEPS=200

export ENH_BASE_LR=8e-6
export ENH_LR_DECAY=0.85
export ENH_BETA=0.15
export ENH_RETAIN_WEIGHT=1.8

export ENH_USE_SIMNPO=1
export ENH_SIMNPO_WEIGHT=0.4
export ENH_SIMNPO_GAMMA=1.0

export ENH_USE_BIAS_NPO=1
export ENH_BIAS_NPO_MARGIN=0.3

export ENH_USE_KL_RETAIN=1
export ENH_KL_RETAIN_WEIGHT=1.5

export ENH_USE_COSINE_LR=1
export ENH_COSINE_LR_MIN_FRAC=0.05

export ENH_USE_SWA=1
export ENH_SWA_START_FRAC=0.4
export ENH_SWA_FREQ=5

export ENH_USE_BS_NPO=0
export ENH_USE_CNPO=0

export ENH_REF_ON_CPU=0
export ENH_USE_DATA_PARALLEL=0

export ENH_USE_EXTERNAL_QA=1
export ENH_EXTERNAL_QA_PATH=/home2/vijay.s/inlp_project/data/harrypotterqa
export ENH_EXTERNAL_QA_MAX=300
export ENH_USE_EXTERNAL_TEXT_FORGET=0
export ENH_TRAIN_FORGET_QA_MAX=300
export ENH_TRAIN_RETAIN_QA_MAX=150

export ENH_HARD_AUGMENT=1
export ENH_HARD_AUG_VARIANTS=3

export ENH_REFUSAL_CALIBRATION=1
export ENH_REFUSAL_HARD_AUGMENT=1
export ENH_REFUSAL_MAX_FORGET=128
export ENH_REFUSAL_MAX_RETAIN=64
export ENH_REFUSAL_EPOCHS=1
export ENH_REFUSAL_TEXT="I can't help with Harry Potter specific details."

export ENH_SAVE_EVERY_STEPS=0
export ENH_CHECKPOINT_DIR=""
export MID_EVAL_STEP=0

export RETRAIN_ENHANCED=1
export SKIP_FULL_EVAL=1
export DQA_ENABLE_ADVERSARIAL=1
export DQA_ADV_MAX_ITEMS=24
export ENH_REPORT_TO=none

bash /home2/vijay.s/inlp_project/remote_sync/run_best_unlearn.sh
