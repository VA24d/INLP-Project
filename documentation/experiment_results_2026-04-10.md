# Experiment Results Summary (2026-04-10)

## Scope

This note captures outcomes for the `rw12b015_tight_gpufix` checkpoint family and adds direct-question probing on top of aggregate evaluator metrics.

Primary artifacts:
- [remote_sync/direct_qa_eval_summary_rw12b015_tight_gpufix.json](../remote_sync/direct_qa_eval_summary_rw12b015_tight_gpufix.json)
- [remote_sync/direct_qa_eval_rw12b015_tight_gpufix.log](../remote_sync/direct_qa_eval_rw12b015_tight_gpufix.log)
- [remote_sync/probes/direct_question_probe_rw12b015_tight_gpufix.md](../remote_sync/probes/direct_question_probe_rw12b015_tight_gpufix.md)
- [remote_sync/probes/direct_question_probe_rw12b015_tight_gpufix.json](../remote_sync/probes/direct_question_probe_rw12b015_tight_gpufix.json)

## Strict + Adversarial Metrics Snapshot

From [remote_sync/direct_qa_eval_summary_rw12b015_tight_gpufix.json](../remote_sync/direct_qa_eval_summary_rw12b015_tight_gpufix.json):

| Model | forget_hit_rate | retain_hit_rate | forget_refusal_success_rate | selection_score | robust_selection_score |
|---|---:|---:|---:|---:|---:|
| Base (FP16) | 0.0833 | 0.8333 | 0.0000 | 0.3750 | 0.2833 |
| Enhanced (FP16) | 0.0000 | 0.5000 | 0.4167 | 0.4542 | 0.4067 |

Adversarial component (same summary file):

| Model | adversarial_forget_hit_rate | adversarial_retain_hit_rate | adversarial_forget_refusal_success_rate |
|---|---:|---:|---:|
| Base (FP16) | 0.0833 | 0.4167 | 0.0000 |
| Enhanced (FP16) | 0.0417 | 0.4167 | 0.2917 |

## Direct Question Probe (Qualitative Layer)

From [remote_sync/probes/direct_question_probe_rw12b015_tight_gpufix.md](../remote_sync/probes/direct_question_probe_rw12b015_tight_gpufix.md):

| Model | Forget QA Hit | Forget Leak | Forget Refusal | Retain QA Hit | Retain Refusal |
|---|---:|---:|---:|---:|---:|
| Base (FP16) | 0.0000 | 0.0000 | 0.0000 | 0.8333 | 0.0000 |
| Enhanced (FP16) | 0.0000 | 0.3333 | 0.6667 | 1.0000 | 0.0000 |

Interpretation:
- Enhanced shows stronger forget-domain refusal behavior than base on this mini probe (`forget_refusal_rate = 0.6667`).
- Enhanced still leaks expected entities in some refusal-style answers (`forget_leak_rate = 0.3333`).
- Retain behavior on this probe is stronger for enhanced (`retain_qa_hit_rate = 1.0000`).

## Notes and Caveats

- Probe set is intentionally small (6 forget + 6 retain) and should be treated as qualitative evidence only.
- Aggregate model selection remains based on strict/adversarial evaluator metrics, especially `robust_selection_score`.
- Direct probe scoring distinguishes:
  - direct correct answer (`qa_hit`),
  - entity leakage (`expected_match`),
  - refusal behavior (`refusal`).
