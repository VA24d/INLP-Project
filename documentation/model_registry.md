# Model Registry

This registry tracks the model variants published during the April 2026 export/upload cycle.

## Hugging Face Repositories

### Legacy enhanced checkpoints

| Variant | Repo |
|---|---|
| enhanced_unlearned | https://huggingface.co/nightbloodredux/enhanced-unlearned |
| enhanced_unlearned_rw08b020 | https://huggingface.co/nightbloodredux/enhanced-unlearned-rw08b020 |
| enhanced_unlearned_rw12b015 | https://huggingface.co/nightbloodredux/enhanced-unlearned-rw12b015 |

### Best-vs-base export bundle

| Variant | Quantization | HF Repo | Local source folder |
|---|---|---|---|
| best_advprobe_r2 | FP16 | https://huggingface.co/nightbloodredux/inlp-best-advprobe-r2-fp16 | model_upload_staging/model_export_bundle/best_fp16 |
| best_advprobe_r2 | INT8 | https://huggingface.co/nightbloodredux/inlp-best-advprobe-r2-int8 | model_upload_staging/model_export_bundle/best_int8 |
| best_advprobe_r2 | INT4 | https://huggingface.co/nightbloodredux/inlp-best-advprobe-r2-int4 | model_upload_staging/model_export_bundle/best_int4 |
| base_gemma3_1b | FP16 | https://huggingface.co/nightbloodredux/inlp-base-gemma3-1b-fp16 | model_upload_staging/model_export_bundle/base_fp16 |
| base_gemma3_1b | INT8 | https://huggingface.co/nightbloodredux/inlp-base-gemma3-1b-int8 | model_upload_staging/model_export_bundle/base_int8 |
| base_gemma3_1b | INT4 | https://huggingface.co/nightbloodredux/inlp-base-gemma3-1b-int4 | model_upload_staging/model_export_bundle/base_int4 |

## Required Files Checklist

Each published model repository should include:
- `model.safetensors`
- `config.json`
- `generation_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `chat_template.jinja`

## Verification Command

Use this to verify all registries in one pass:

```bash
python - <<'PY'
from huggingface_hub import HfApi

repos = [
    'nightbloodredux/enhanced-unlearned',
    'nightbloodredux/enhanced-unlearned-rw08b020',
    'nightbloodredux/enhanced-unlearned-rw12b015',
    'nightbloodredux/inlp-best-advprobe-r2-fp16',
    'nightbloodredux/inlp-best-advprobe-r2-int8',
    'nightbloodredux/inlp-best-advprobe-r2-int4',
    'nightbloodredux/inlp-base-gemma3-1b-fp16',
    'nightbloodredux/inlp-base-gemma3-1b-int8',
    'nightbloodredux/inlp-base-gemma3-1b-int4',
]

required = {
    'model.safetensors',
    'config.json',
    'generation_config.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'chat_template.jinja',
}

api = HfApi()
for repo in repos:
    files = set(api.list_repo_files(repo, repo_type='model'))
    missing = sorted(required - files)
    print(repo, 'OK' if not missing else f'MISSING {missing}')
PY
```
