#!/usr/bin/env bash
set -euo pipefail

cd /home/bhaskar/inlp

PY="/home/bhaskar/aise/venvs/phase2/bin/python"

export HF_HOME=/tmp/hf_cache
export TRANSFORMERS_CACHE=/tmp/hf_cache
export HF_DATASETS_CACHE=/tmp/hf_cache/datasets
export HF_HUB_DISABLE_XET=1

mkdir -p /tmp/hf_cache /tmp/hf_cache/datasets

run_one() {
  local tag="$1"
  local beta="$2"
  local retain_weight="$3"
  local ckpt="/home/bhaskar/inlp/models_enhanced/enhanced_unlearned_${tag}"

  echo "[START] tag=${tag} beta=${beta} retain_weight=${retain_weight}"

  ENH_EXPERIMENT="$tag" \
  RETRAIN_ENHANCED=1 \
  REQUANTIZE=0 \
    SKIP_FULL_EVAL=1 \
  ENH_TOTAL_PHASES=2 \
  ENH_EPOCHS_PER_PHASE=1 \
    ENH_MAX_STEPS=120 \
  ENH_BETA="$beta" \
  ENH_RETAIN_WEIGHT="$retain_weight" \
  "$PY" run_enhanced.py > "run_${tag}.log" 2>&1

  ENH_EXPERIMENT="$tag" \
  ENHANCED_PATH="$ckpt" \
    RUN_4BIT="${RUN_4BIT:-0}" \
  "$PY" direct_qa_eval.py > "direct_qa_eval_${tag}.log" 2>&1

  echo "[DONE] tag=${tag}"
}

# Sweep variants around current default to trade off forgetting vs retention.
run_one "rw08b020" "0.20" "0.80"
run_one "rw12b015" "0.15" "1.20"

"$PY" - <<'PY'
import csv
import glob
import json
import os

rows = []
for p in sorted(glob.glob('/home/bhaskar/inlp/direct_qa_eval_summary*.json')):
    tag = os.path.basename(p).replace('direct_qa_eval_summary_', '').replace('.json', '')
    if tag == 'direct_qa_eval_summary':
        tag = 'default'
    with open(p, 'r') as f:
        data = json.load(f)

    for model, m in data.items():
        if not model.startswith('Enhanced'):
            continue
        forget = float(m.get('forget_hit_rate', 1.0))
        retain = float(m.get('retain_hit_rate', 0.0))
        forget_refusal_success = float(m.get('forget_refusal_success_rate', 0.0))
        retain_refusal_rate = float(m.get('retain_refusal_rate', 0.0))
        avg_noise = float(m.get('avg_noise', 0.0))
        score = float(m.get('selection_score', ((1.0 - forget) + retain)))
        rows.append({
            'tag': tag,
            'model': model,
            'forget_hit_rate': forget,
            'retain_hit_rate': retain,
            'forget_refusal_success_rate': forget_refusal_success,
            'retain_refusal_rate': retain_refusal_rate,
            'avg_noise': avg_noise,
            'selection_score': score,
        })

rows.sort(key=lambda r: (-r['selection_score'], r['forget_hit_rate'], -r['retain_hit_rate']))

out_csv = '/home/bhaskar/inlp/experiment_scoreboard.csv'
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            'tag',
            'model',
            'forget_hit_rate',
            'retain_hit_rate',
            'forget_refusal_success_rate',
            'retain_refusal_rate',
            'avg_noise',
            'selection_score',
        ],
    )
    w.writeheader()
    w.writerows(rows)

print('Saved', out_csv)
for r in rows[:10]:
    print(r)
PY

echo "[ALL DONE]"
