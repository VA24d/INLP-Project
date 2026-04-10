#!/usr/bin/env bash
set -euo pipefail

# Run the best-performing unlearning profile in one command.
#
# Defaults are based on the strongest current profile from this repo's sweeps
# (rw12b015-style settings), but every knob remains overridable via env vars.

ROOT_DIR="${ROOT_DIR:-/home/bhaskar/inlp}"
cd "$ROOT_DIR"

if [[ -x "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
elif [[ -x "/home/bhaskar/aise/venvs/phase2/bin/python" ]]; then
  PY="/home/bhaskar/aise/venvs/phase2/bin/python"
elif [[ -x "/usr/local/bin/python3.9" ]]; then
  PY="/usr/local/bin/python3.9"
else
  PY="python3"
fi

RUN_ENHANCED_SCRIPT="remote_sync/run_enhanced.py"
DIRECT_QA_SCRIPT="remote_sync/direct_qa_eval.py"
if [[ ! -f "$RUN_ENHANCED_SCRIPT" ]]; then
  RUN_ENHANCED_SCRIPT="run_enhanced.py"
fi
if [[ ! -f "$DIRECT_QA_SCRIPT" ]]; then
  DIRECT_QA_SCRIPT="direct_qa_eval.py"
fi

export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/tmp/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf_cache/datasets}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

# Best-profile defaults (overridable)
TAG="${ENH_EXPERIMENT:-rw12b015}"
BETA="${ENH_BETA:-0.15}"
RETAIN_WEIGHT="${ENH_RETAIN_WEIGHT:-1.20}"
USE_BS_NPO="${ENH_USE_BS_NPO:-0}"
BS_NPO_WEIGHT="${ENH_BS_NPO_WEIGHT:-0.35}"
BS_NPO_START_STEP="${ENH_BS_NPO_START_STEP:-0}"
BS_NPO_INTERVAL="${ENH_BS_NPO_INTERVAL:-1}"
USE_CNPO="${ENH_USE_CNPO:-0}"
CNPO_WEIGHT="${ENH_CNPO_WEIGHT:-0.25}"
CNPO_START_STEP="${ENH_CNPO_START_STEP:-0}"
CNPO_INTERVAL="${ENH_CNPO_INTERVAL:-1}"
CNPO_TEMP="${ENH_CNPO_TEMP:-6.0}"
CNPO_RETAIN_PULL="${ENH_CNPO_RETAIN_PULL:-0.6}"
TOTAL_PHASES="${ENH_TOTAL_PHASES:-2}"
EPOCHS_PER_PHASE="${ENH_EPOCHS_PER_PHASE:-1}"
MAX_STEPS="${ENH_MAX_STEPS:-120}"
HARD_AUGMENT="${ENH_HARD_AUGMENT:-1}"
HARD_AUG_VARIANTS="${ENH_HARD_AUG_VARIANTS:-7}"
REFUSAL_HARD_AUGMENT="${ENH_REFUSAL_HARD_AUGMENT:-1}"
USE_EXTERNAL_QA="${ENH_USE_EXTERNAL_QA:-0}"
EXTERNAL_QA_PATH="${ENH_EXTERNAL_QA_PATH:-/home/bhaskar/inlp/data/harrypotterqa}"
EXTERNAL_QA_MAX="${ENH_EXTERNAL_QA_MAX:-300}"
USE_EXTERNAL_TEXT_FORGET="${ENH_USE_EXTERNAL_TEXT_FORGET:-0}"
EXTERNAL_TEXT_PATH="${ENH_EXTERNAL_TEXT_PATH:-}"
EXTERNAL_TEXT_KAGGLE_DATASET="${ENH_EXTERNAL_TEXT_KAGGLE_DATASET:-}"
EXTERNAL_TEXT_MAX="${ENH_EXTERNAL_TEXT_MAX:-400}"
EXTERNAL_TEXT_CLOZE_MAX="${ENH_EXTERNAL_TEXT_CLOZE_MAX:-200}"
REF_ON_CPU="${ENH_REF_ON_CPU:-1}"
REF_CPU_DTYPE="${ENH_REF_CPU_DTYPE:-float16}"
DATALOADER_WORKERS="${ENH_DATALOADER_WORKERS:-0}"
TRAINER_DATALOADER_WORKERS="${ENH_TRAINER_DATALOADER_WORKERS:-0}"
MAP_NUM_PROC="${ENH_MAP_NUM_PROC:-0}"
TOKENIZE_INPUT_MAX_MULT="${ENH_TOKENIZE_INPUT_MAX_MULT:-1}"
USE_DATA_PARALLEL="${ENH_USE_DATA_PARALLEL:-0}"
SAVE_EVERY_STEPS="${ENH_SAVE_EVERY_STEPS:-0}"
MID_CKPT_DIR_DEFAULT="$ROOT_DIR/models_enhanced/midcheckpoints_${TAG}"
MID_CKPT_DIR="${ENH_CHECKPOINT_DIR:-$MID_CKPT_DIR_DEFAULT}"
MID_EVAL_STEP="${MID_EVAL_STEP:-0}"

RETRAIN="${RETRAIN_ENHANCED:-1}"
RUN_4BIT="${RUN_4BIT:-0}"
SKIP_FULL_EVAL="${SKIP_FULL_EVAL:-1}"
REQ="${REQUANTIZE:-0}"
DQA_ENABLE_ADVERSARIAL="${DQA_ENABLE_ADVERSARIAL:-1}"

CKPT_DEFAULT="$ROOT_DIR/models_enhanced/enhanced_unlearned_${TAG}"
CKPT_PATH="${ENHANCED_CKPT_PATH:-$CKPT_DEFAULT}"

RUN_LOG="run_best_${TAG}.log"
EVAL_LOG="direct_qa_eval_best_${TAG}.log"
SUMMARY_PATH="${DIRECT_QA_SUMMARY_PATH:-$ROOT_DIR/direct_qa_eval_summary_${TAG}_best.json}"
export DIRECT_QA_SUMMARY_PATH="$SUMMARY_PATH"

echo "[INFO] ROOT_DIR=$ROOT_DIR"
echo "[INFO] PY=$PY"
echo "[INFO] RUN_ENHANCED_SCRIPT=$RUN_ENHANCED_SCRIPT"
echo "[INFO] DIRECT_QA_SCRIPT=$DIRECT_QA_SCRIPT"
echo "[INFO] TAG=$TAG CKPT_PATH=$CKPT_PATH"
echo "[INFO] HP: beta=$BETA retain_weight=$RETAIN_WEIGHT phases=$TOTAL_PHASES epochs_per_phase=$EPOCHS_PER_PHASE max_steps=$MAX_STEPS"
echo "[INFO] BS-NPO: enabled=$USE_BS_NPO weight=$BS_NPO_WEIGHT start_step=$BS_NPO_START_STEP interval=$BS_NPO_INTERVAL"
echo "[INFO] CNPO: enabled=$USE_CNPO weight=$CNPO_WEIGHT start_step=$CNPO_START_STEP interval=$CNPO_INTERVAL temp=$CNPO_TEMP retain_pull=$CNPO_RETAIN_PULL"
echo "[INFO] EXT_QA: use=$USE_EXTERNAL_QA path=$EXTERNAL_QA_PATH max=$EXTERNAL_QA_MAX"
echo "[INFO] HARD: augment=$HARD_AUGMENT variants=$HARD_AUG_VARIANTS refusal_hard_augment=$REFUSAL_HARD_AUGMENT"
echo "[INFO] EXT_TEXT: use=$USE_EXTERNAL_TEXT_FORGET path=$EXTERNAL_TEXT_PATH kaggle_ref=$EXTERNAL_TEXT_KAGGLE_DATASET max=$EXTERNAL_TEXT_MAX cloze_max=$EXTERNAL_TEXT_CLOZE_MAX"
echo "[INFO] PERF: ref_on_cpu=$REF_ON_CPU ref_cpu_dtype=$REF_CPU_DTYPE dl_workers=$DATALOADER_WORKERS trainer_workers=$TRAINER_DATALOADER_WORKERS map_num_proc=$MAP_NUM_PROC token_mult=$TOKENIZE_INPUT_MAX_MULT"
echo "[INFO] PARALLEL: use_data_parallel=$USE_DATA_PARALLEL save_every_steps=$SAVE_EVERY_STEPS mid_ckpt_dir=$MID_CKPT_DIR mid_eval_step=$MID_EVAL_STEP"
echo "[INFO] FLAGS: retrain=$RETRAIN skip_full_eval=$SKIP_FULL_EVAL requantize=$REQ run_4bit=$RUN_4BIT adv_eval=$DQA_ENABLE_ADVERSARIAL"

echo "[STEP] Unlearning run..."
ENH_EXPERIMENT="$TAG" \
ENHANCED_CKPT_PATH="$CKPT_PATH" \
RETRAIN_ENHANCED="$RETRAIN" \
REQUANTIZE="$REQ" \
SKIP_FULL_EVAL="$SKIP_FULL_EVAL" \
ENH_TOTAL_PHASES="$TOTAL_PHASES" \
ENH_EPOCHS_PER_PHASE="$EPOCHS_PER_PHASE" \
ENH_MAX_STEPS="$MAX_STEPS" \
ENH_BETA="$BETA" \
ENH_RETAIN_WEIGHT="$RETAIN_WEIGHT" \
ENH_USE_BS_NPO="$USE_BS_NPO" \
ENH_BS_NPO_WEIGHT="$BS_NPO_WEIGHT" \
ENH_BS_NPO_START_STEP="$BS_NPO_START_STEP" \
ENH_BS_NPO_INTERVAL="$BS_NPO_INTERVAL" \
ENH_USE_CNPO="$USE_CNPO" \
ENH_CNPO_WEIGHT="$CNPO_WEIGHT" \
ENH_CNPO_START_STEP="$CNPO_START_STEP" \
ENH_CNPO_INTERVAL="$CNPO_INTERVAL" \
ENH_CNPO_TEMP="$CNPO_TEMP" \
ENH_CNPO_RETAIN_PULL="$CNPO_RETAIN_PULL" \
ENH_HARD_AUGMENT="$HARD_AUGMENT" \
ENH_HARD_AUG_VARIANTS="$HARD_AUG_VARIANTS" \
ENH_REFUSAL_HARD_AUGMENT="$REFUSAL_HARD_AUGMENT" \
ENH_USE_EXTERNAL_QA="$USE_EXTERNAL_QA" \
ENH_EXTERNAL_QA_PATH="$EXTERNAL_QA_PATH" \
ENH_EXTERNAL_QA_MAX="$EXTERNAL_QA_MAX" \
ENH_USE_EXTERNAL_TEXT_FORGET="$USE_EXTERNAL_TEXT_FORGET" \
ENH_EXTERNAL_TEXT_PATH="$EXTERNAL_TEXT_PATH" \
ENH_EXTERNAL_TEXT_KAGGLE_DATASET="$EXTERNAL_TEXT_KAGGLE_DATASET" \
ENH_EXTERNAL_TEXT_MAX="$EXTERNAL_TEXT_MAX" \
ENH_EXTERNAL_TEXT_CLOZE_MAX="$EXTERNAL_TEXT_CLOZE_MAX" \
ENH_REF_ON_CPU="$REF_ON_CPU" \
ENH_REF_CPU_DTYPE="$REF_CPU_DTYPE" \
ENH_DATALOADER_WORKERS="$DATALOADER_WORKERS" \
ENH_TRAINER_DATALOADER_WORKERS="$TRAINER_DATALOADER_WORKERS" \
ENH_MAP_NUM_PROC="$MAP_NUM_PROC" \
ENH_TOKENIZE_INPUT_MAX_MULT="$TOKENIZE_INPUT_MAX_MULT" \
ENH_USE_DATA_PARALLEL="$USE_DATA_PARALLEL" \
ENH_SAVE_EVERY_STEPS="$SAVE_EVERY_STEPS" \
ENH_CHECKPOINT_DIR="$MID_CKPT_DIR" \
"$PY" -u "$RUN_ENHANCED_SCRIPT" > "$RUN_LOG" 2>&1
echo "[DONE] Unlearning complete -> $RUN_LOG"

if [[ "$MID_EVAL_STEP" =~ ^[0-9]+$ ]] && [[ "$MID_EVAL_STEP" -gt 0 ]]; then
  MID_STEP_FMT="$(printf "%04d" "$MID_EVAL_STEP")"
  MID_PATH="$MID_CKPT_DIR/step_${MID_STEP_FMT}"
  MID_SUMMARY_PATH="${ROOT_DIR}/direct_qa_eval_summary_${TAG}_mid_s${MID_STEP_FMT}.json"
  MID_EVAL_LOG="direct_qa_eval_mid_${TAG}_s${MID_STEP_FMT}.log"
  if [[ -d "$MID_PATH" ]]; then
    echo "[STEP] Middle checkpoint eval at step ${MID_STEP_FMT}..."
    ENH_EXPERIMENT="${TAG}_mid_s${MID_STEP_FMT}" \
    ENHANCED_PATH="$MID_PATH" \
    RUN_4BIT="0" \
    DQA_ENABLE_ADVERSARIAL="$DQA_ENABLE_ADVERSARIAL" \
    DIRECT_QA_SUMMARY_PATH="$MID_SUMMARY_PATH" \
    "$PY" -u "$DIRECT_QA_SCRIPT" > "$MID_EVAL_LOG" 2>&1
    echo "[DONE] Middle eval complete -> $MID_EVAL_LOG"
    echo "[INFO] Middle summary -> $MID_SUMMARY_PATH"
  else
    echo "[WARN] Requested middle checkpoint not found: $MID_PATH"
  fi
fi

echo "[STEP] Direct QA eval..."
ENH_EXPERIMENT="$TAG" \
ENHANCED_PATH="$CKPT_PATH" \
RUN_4BIT="$RUN_4BIT" \
DQA_ENABLE_ADVERSARIAL="$DQA_ENABLE_ADVERSARIAL" \
DIRECT_QA_SUMMARY_PATH="$SUMMARY_PATH" \
"$PY" -u "$DIRECT_QA_SCRIPT" > "$EVAL_LOG" 2>&1
echo "[DONE] Eval complete -> $EVAL_LOG"

echo "[STEP] Summary snapshot..."
"$PY" - <<'PY'
import json
import os
import sys

path = os.environ["DIRECT_QA_SUMMARY_PATH"]
if not os.path.exists(path):
    print(f"[WARN] Summary file not found: {path}")
    sys.exit(0)

with open(path, "r") as f:
    data = json.load(f)

print(f"[INFO] Summary file: {path}")
for model_name, metrics in data.items():
    status = metrics.get("status", "ok")
    if status != "ok":
        print(f"  - {model_name}: status={status} error={metrics.get('error')}")
        continue
    print(
        "  - "
        f"{model_name}: "
        f"forget_hit_rate={metrics.get('forget_hit_rate')} "
        f"retain_hit_rate={metrics.get('retain_hit_rate')} "
        f"selection_score={metrics.get('selection_score')} "
        f"robust_selection_score={metrics.get('robust_selection_score')}"
    )
PY

echo "[ALL DONE]"
