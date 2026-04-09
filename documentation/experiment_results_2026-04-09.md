# Experiment Results Summary (2026-04-09)

## Scope

This document summarizes the main outcomes from the remote enhanced unlearning sweeps and adversarial probing experiments captured under [remote_sync](../remote_sync).

Primary result artifacts:
- [remote_sync/direct_qa_adv_scoreboard.csv](../remote_sync/direct_qa_adv_scoreboard.csv)
- [remote_sync/direct_qa_eval_summary_advprobe_r1.json](../remote_sync/direct_qa_eval_summary_advprobe_r1.json)
- [remote_sync/direct_qa_eval_summary_advprobe_r2.json](../remote_sync/direct_qa_eval_summary_advprobe_r2.json)
- [remote_sync/direct_qa_eval_summary_advprobe_r3.json](../remote_sync/direct_qa_eval_summary_advprobe_r3.json)

## Best Run

Best enhanced checkpoint by robust metric:
- Experiment tag: `advprobe_r2`
- Model: `Enhanced (FP16)`
- `robust_selection_score`: `0.7175`
- `selection_score`: `0.75`

### advprobe_r2 (Enhanced FP16)

| Metric | Value |
|---|---:|
| forget_hit_rate | 0.0 |
| retain_hit_rate | 0.75 |
| forget_refusal_success_rate | 0.75 |
| retain_refusal_rate | 0.0 |
| avg_noise | 0.0 |
| selection_score | 0.75 |
| adversarial_forget_hit_rate | 0.0 |
| adversarial_retain_hit_rate | 0.75 |
| adversarial_forget_refusal_success_rate | 0.625 |
| adversarial_retain_refusal_rate | 0.0 |
| adversarial_avg_noise | 0.0 |
| robust_selection_score | 0.7175 |

Source: [remote_sync/direct_qa_eval_summary_advprobe_r2.json](../remote_sync/direct_qa_eval_summary_advprobe_r2.json)

## Ranking Snapshot (Enhanced FP16)

From [remote_sync/direct_qa_adv_scoreboard.csv](../remote_sync/direct_qa_adv_scoreboard.csv):

| Rank | Tag | robust_selection_score | selection_score |
|---:|---|---:|---:|
| 1 | advprobe_r2 | 0.7175 | 0.75 |
| 2 | advprobe_r3 | 0.5875 | 0.6041666667 |
| 3 | advprobe_r1 | 0.4950 | 0.5125 |
| 4 | default | 0.4100 | 0.45 |

## Base vs Best Enhanced (Robust Metric)

Using rows for tag `advprobe_r2` in [remote_sync/direct_qa_adv_scoreboard.csv](../remote_sync/direct_qa_adv_scoreboard.csv):

- Base (FP16) robust score: `0.3475`
- Enhanced (FP16, advprobe_r2) robust score: `0.7175`
- Absolute gain: `+0.3700`

## Notes

- The strict/adversarial evaluator emits both standard and adversarial metrics.
- Summaries and logs for additional sweeps (`fixbal_q1`, `fixbal_q2`, `qaextquick`) are preserved under [remote_sync](../remote_sync).
- Published model artifacts for best/base variants are tracked in [documentation/model_registry.md](model_registry.md).
