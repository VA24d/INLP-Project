# Handover: Supplemented Unlearning Run (2026-04-10)

## 1) Scope of This Handover
This handover covers the latest remote run sequence for the enhanced unlearning pipeline with:

- external forget-text supplementation from Kaggle HarryPotterKB,
- generated cloze forget-QA from that text,
- memory-safe settings for low-RAM execution,
- adversarial direct-QA evaluation enabled.

Main objective during this cycle was to make the supplemented run stable and then assess quality.

## 2) What Changed in Code

Primary files touched in this workstream:

- `remote_sync/run_enhanced.py`
  - Added external text ingestion and parsing helpers.
  - Added generated cloze QA creation from external text.
  - Added env flags for external text path / Kaggle dataset / caps.
  - Merged external forget rows and cloze QA into training signal.
- `remote_sync/run_best_unlearn.sh`
  - Added pass-through env wiring for external text and cloze controls.
  - Added run-time logging for external text settings.
- `remote_sync/direct_qa_eval.py`
  - Uses generated adversarial perturbation rows during eval (already integrated in prior step).

## 3) Run Attempts and Root Cause

### Failure mode observed
Earlier supplemented run failed with CPU OOM during SWA update:

`RuntimeError: DefaultCPUAllocator: can't allocate memory: you tried to allocate 1207959552 bytes`

This occurred in SWA averaging (`swa.update(model)`).

### Stability fix applied
Successful run used:

- `ENH_USE_SWA=0` (critical for avoiding CPU RAM blow-up),
- low-memory settings (`ENH_BATCH_SIZE=1`, `ENH_MAX_SEQ_LEN=96`, worker counts at 0),
- CPU reference model in fp16 (`ENH_REF_ON_CPU=1`, `ENH_REF_CPU_DTYPE=float16`),
- conservative augmentation caps.

## 4) Final Run Configuration (Successful)

Key env profile from final successful run:

- `ENH_USE_EXTERNAL_TEXT_FORGET=1`
- `ENH_EXTERNAL_TEXT_KAGGLE_DATASET=pratikshaaigal/harrypotterkb`
- `ENH_EXTERNAL_TEXT_MAX=180`
- `ENH_EXTERNAL_TEXT_CLOZE_MAX=80`
- `ENH_USE_SWA=0`
- `ENH_MAX_STEPS=40`
- `ENH_GRAD_ACCUM=4`
- `ENH_HARD_AUGMENT=1`
- `ENH_HARD_AUG_VARIANTS=2`
- `ENH_REFUSAL_HARD_AUGMENT=1`
- `ENH_REFUSAL_MAX_FORGET=48`
- `ENH_REFUSAL_MAX_RETAIN=24`
- `ENH_REFUSAL_EPOCHS=1`
- `DQA_ENABLE_ADVERSARIAL=1`
- `DQA_ADV_MAX_ITEMS=12`

Observed data expansion in successful run:

- External text snippets loaded: `180`
- Generated cloze QA: `80`
- Train signal expanded: forget `4 -> 424`, retain `25 -> 137`

## 5) Current Status

- Run completed successfully (no active remote training/eval process).
- Artifacts were generated for this run.
- Stability issue is resolved for this configuration.
- Quality remains below target for unlearning behavior.

## 6) Metrics Snapshot (Latest)

Source:
- `remote_sync/direct_qa_eval_summary_rw12b015_best.json`

### Base (FP16)
- `selection_score`: `0.375`
- `robust_selection_score`: `0.2775`
- `adversarial_selection_score`: `0.13125`
- `forget_hit_rate`: `0.08333`
- `forget_refusal_success_rate`: `0.0`
- `retain_hit_rate`: `0.83333`
- `retain_refusal_rate`: `0.0`

### Enhanced (FP16)
- `selection_score`: `0.3375`
- `robust_selection_score`: `0.255`
- `adversarial_selection_score`: `0.13125`
- `forget_hit_rate`: `0.08333`
- `forget_refusal_success_rate`: `0.0`
- `retain_hit_rate`: `0.75`
- `retain_refusal_rate`: `0.0`

Interpretation:

- Enhanced is currently not outperforming Base on the main selection metrics.
- Forget-side refusal behavior is still weak (`forget_refusal_success_rate=0.0`).
- Retain knowledge is reasonably preserved but with some drop vs Base.

## 7) Artifact Locations

Local artifacts:

- `remote_sync/run_best_rw12b015.log`
- `remote_sync/direct_qa_eval_best_rw12b015.log`
- `remote_sync/direct_qa_eval_summary_rw12b015_best.json`
- `remote_sync/enhanced_unlearning_results_rw12b015.csv`

Remote artifacts (host `/tmp`):

- `/tmp/run_best_rw12b015.log`
- `/tmp/direct_qa_eval_best_rw12b015.log`
- `/tmp/direct_qa_eval_summary_rw12b015_best.json`
- `/tmp/enhanced_unlearning_results_rw12b015.csv`

## 8) Recommended Next Actions

1. Improve forget refusal behavior (primary gap)
- Increase refusal calibration depth and targeted forget prompts.
- Raise refusal weight or apply stricter refusal-trigger templates.

2. Add richer metrics in eval output
- Implement requested lexical/overlap metrics (ROUGE/BLEU/EM/token-F1) in `remote_sync/direct_qa_eval.py` summary payload.

3. Run a small controlled sweep around current stable profile
- Keep `ENH_USE_SWA=0`.
- Sweep `ENH_HARD_AUG_VARIANTS`, refusal caps, and retain weight.
- Track deltas in `robust_selection_score` and `adversarial_selection_score`.

4. Optional robustness checks
- Expand adversarial sample count above 12 for less noisy signal.
- Evaluate whether cloze generation quality needs filtering before training.

## 9) Repro Command Template (Sanitized)

Use this pattern from repo root (replace placeholders):

```bash
sshpass -p '<password>' scp -o StrictHostKeyChecking=accept-new \
  remote_sync/run_enhanced.py remote_sync/direct_qa_eval.py remote_sync/run_best_unlearn.sh \
  root@<host>:/tmp/

sshpass -p '<password>' ssh -o StrictHostKeyChecking=accept-new root@<host> '
  set -e
  cd /tmp
  chmod +x run_best_unlearn.sh
  ROOT_DIR=/tmp \
  PYTHON_BIN=/usr/local/bin/python3.9 \
  MODEL_NAME=nightbloodredux/inlp-base-gemma3-1b-fp16 \
  BASE_MODEL_PATH=nightbloodredux/inlp-base-gemma3-1b-fp16 \
  ENH_USE_EXTERNAL_TEXT_FORGET=1 \
  ENH_EXTERNAL_TEXT_KAGGLE_DATASET=pratikshaaigal/harrypotterkb \
  ENH_EXTERNAL_TEXT_MAX=180 \
  ENH_EXTERNAL_TEXT_CLOZE_MAX=80 \
  ENH_DATALOADER_WORKERS=0 \
  ENH_TRAINER_DATALOADER_WORKERS=0 \
  ENH_MAP_NUM_PROC=0 \
  ENH_TOKENIZE_INPUT_MAX_MULT=1 \
  ENH_BATCH_SIZE=1 \
  ENH_MAX_SEQ_LEN=96 \
  ENH_REF_ON_CPU=1 \
  ENH_REF_CPU_DTYPE=float16 \
  ENH_TOTAL_PHASES=1 \
  ENH_EPOCHS_PER_PHASE=1 \
  ENH_MAX_STEPS=40 \
  ENH_GRAD_ACCUM=4 \
  ENH_USE_SWA=0 \
  ENH_HARD_AUGMENT=1 \
  ENH_HARD_AUG_VARIANTS=2 \
  ENH_REFUSAL_HARD_AUGMENT=1 \
  ENH_REFUSAL_MAX_FORGET=48 \
  ENH_REFUSAL_MAX_RETAIN=24 \
  ENH_REFUSAL_EPOCHS=1 \
  ENH_REPORT_TO=none \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  DQA_ENABLE_ADVERSARIAL=1 \
  DQA_ADV_MAX_ITEMS=12 \
  bash /tmp/run_best_unlearn.sh
'
```

---

Owner handoff note: current baseline for stable supplemented execution is now documented and reproducible; priority should shift from runtime stability to quality tuning and richer evaluation metrics.