# INLP Project: Machine Unlearning with Gemma

Iterative unlearning experiments on Google Gemma 3 1B IT with a focus on removing targeted knowledge while preserving general behavior under strict and adversarial evaluation.

## Overview

This repository contains:
- Modular unlearning/evaluation scripts under [scripts](scripts)
- Kaggle notebooks under [kaggle](kaggle)
- Extended remote experiment orchestration and evaluator logic under [remote_sync](remote_sync)
- A browser-based interactive demo under [docs](docs)
- Execution and handover documentation under [documentation](documentation)

Core methods used in this workstream:
- Task arithmetic / task-vector style unlearning
- Gradient-ascent based forgetting pressure
- Refusal calibration for forget-domain questions
- Quantized exports (FP16, INT8, INT4)
- Adversarial probing and robust scoring

## April 2026 Status Snapshot

Latest experiment batch includes `advprobe_r1/r2/r3`, `fixbal_q1/q2`, and `qaextquick`.

Top robust run from [remote_sync/direct_qa_adv_scoreboard.csv](remote_sync/direct_qa_adv_scoreboard.csv):
- Tag: `advprobe_r2`
- Model: `Enhanced (FP16)`
- `robust_selection_score`: `0.7175`
- `selection_score`: `0.75`

All planned Hugging Face model pushes are complete and verified.

## Model Registry (Published)

### Earlier enhanced checkpoints
- https://huggingface.co/nightbloodredux/enhanced-unlearned
- https://huggingface.co/nightbloodredux/enhanced-unlearned-rw08b020
- https://huggingface.co/nightbloodredux/enhanced-unlearned-rw12b015

### Best-vs-base export bundle (FP16/INT8/INT4)
- https://huggingface.co/nightbloodredux/inlp-best-advprobe-r2-fp16
- https://huggingface.co/nightbloodredux/inlp-best-advprobe-r2-int8
- https://huggingface.co/nightbloodredux/inlp-best-advprobe-r2-int4
- https://huggingface.co/nightbloodredux/inlp-base-gemma3-1b-fp16
- https://huggingface.co/nightbloodredux/inlp-base-gemma3-1b-int8
- https://huggingface.co/nightbloodredux/inlp-base-gemma3-1b-int4

See [documentation/model_registry.md](documentation/model_registry.md) for a per-variant table and verification method.

## Repository Structure

```text
.
├── docs/                        # GitHub Pages / interactive frontend assets
├── documentation/               # Guides, handover notes, reports
├── kaggle/                      # Notebook workflows
├── model_upload_staging/        # Local model export/upload staging (gitignored)
├── muse_bench/                  # Benchmark framework
├── remote_sync/                 # Remote-run scripts, logs, summaries, scoreboards
└── scripts/                     # Modular baseline unlearning pipeline
```

## Quick Start

### 1) Local website
```bash
python3 -m http.server 8080 --directory docs
```
Then open `http://localhost:8080`.

### 2) Baseline modular pipeline
```bash
python scripts/01_load_dataset_model.py
python scripts/02_task_arithmetic_unlearning.py
python scripts/03_gradient_ascent_unlearning.py
python scripts/04_quantize_models.py
python scripts/05_evaluation.py
```

### 3) Remote enhanced sweep/eval workflow
Use [remote_sync/run_enhanced.py](remote_sync/run_enhanced.py), [remote_sync/direct_qa_eval.py](remote_sync/direct_qa_eval.py), and [remote_sync/run_sweep_experiments.sh](remote_sync/run_sweep_experiments.sh).

## Documentation Index

- Execution guide: [documentation/execution_guide.md](documentation/execution_guide.md)
- Result summary (April 9, 2026): [documentation/experiment_results_2026-04-09.md](documentation/experiment_results_2026-04-09.md)
- Model registry and verification: [documentation/model_registry.md](documentation/model_registry.md)
- Upload/sync handover log: [documentation/handover_upload_and_sync_2026-04-09.md](documentation/handover_upload_and_sync_2026-04-09.md)

## Security Note

Credentials must be provided via environment variables (for example `HF_TOKEN`, `WANDB_API_KEY`) or secret managers. Do not hardcode tokens in notebooks/scripts.

## Team

- Vijay
- Anurag
- Aryanil
- Harsh

Powered by PyTorch, Hugging Face, and WebLLM.
