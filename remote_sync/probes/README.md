# Probes Folder

This folder stores direct-question qualitative probes that complement `remote_sync/direct_qa_eval.py` aggregate metrics.

## Files

- `direct_question_probe.py`: Runs a short QA probe against two local model paths (base and enhanced).
- `direct_question_probe_rw12b015_tight_gpufix.json`: Machine-readable output for the `rw12b015_tight_gpufix` run.
- `direct_question_probe_rw12b015_tight_gpufix.md`: Human-readable table report for the same run.

## Why this exists

Aggregate metrics can hide behavior patterns. These probe outputs expose concrete model answers, refusal patterns, and potential leakage in forget-domain prompts.

## Run command

From repository root:

```bash
/opt/anaconda3/bin/conda run -p /opt/anaconda3 --no-capture-output python remote_sync/probes/direct_question_probe.py \
  --tag rw12b015_tight_gpufix \
  --base-model model_upload_staging/base_gemma3_1b_it_fp16 \
  --enhanced-model model_upload_staging/enhanced_unlearned_rw12b015_tight_gpufix \
  --out-json remote_sync/probes/direct_question_probe_rw12b015_tight_gpufix.json \
  --out-md remote_sync/probes/direct_question_probe_rw12b015_tight_gpufix.md
```

## Headline metrics in probe output

- `forget_qa_hit_rate`: Fraction of forget prompts with a direct correct answer (refusal excluded).
- `forget_leak_rate`: Fraction of forget prompts where expected entity appears anywhere in output.
- `forget_refusal_rate`: Fraction of forget prompts that show refusal language.
- `retain_qa_hit_rate`: Fraction of retain prompts answered correctly.
- `retain_refusal_rate`: Fraction of retain prompts refused.
