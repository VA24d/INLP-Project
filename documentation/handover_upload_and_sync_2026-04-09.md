# Handover: Sync + Export + Upload Workstream (2026-04-09)

## 1) What Was Happening
The workflow looked stuck for three concrete reasons:

1. Local rsync version mismatch
- The local macOS rsync does not support these flags:
  - --ignore-missing-args
  - --append-verify
- Commands using those flags exited immediately and had to be rewritten.

2. Long-copy SSH disconnects
- Large model files (about 1 to 2 GB each) caused intermittent SSH broken pipe disconnects from remote.
- Transfer needed resumable rsync mode with retries.

3. Tool timeout/background behavior
- One long rsync command exceeded tool timeout and was moved to a background terminal.
- Output looked paused because the tool was not attached to live completion.

## 2) Current State

### Remote export bundle
Path:
- /home/bhaskar/inlp/model_export_bundle

Contains 6 folders:
- best_fp16
- best_int8
- best_int4
- base_fp16
- base_int8
- base_int4

### Local export bundle
Path:
- model_upload_staging/model_export_bundle

Contains all 6 folders and model.safetensors files for each variant.

### Evaluation artifacts
Synced locally under:
- remote_sync/

Notable files:
- direct_qa_eval_summary_advprobe_r1.json
- direct_qa_eval_summary_advprobe_r2.json
- direct_qa_eval_summary_advprobe_r3.json
- direct_qa_adv_scoreboard.csv

Best model by robust score:
- tag: advprobe_r2
- robust_selection_score: 0.7175
- source: remote_sync/direct_qa_adv_scoreboard.csv

## 3) What Is Still Pending
Upload of the new 6 export variants to Hugging Face repos is still pending.

## 4) Recommended Repo Mapping
Use one repo per variant for clarity:

- best_fp16  -> nightbloodredux/inlp-best-advprobe-r2-fp16
- best_int8  -> nightbloodredux/inlp-best-advprobe-r2-int8
- best_int4  -> nightbloodredux/inlp-best-advprobe-r2-int4
- base_fp16  -> nightbloodredux/inlp-base-gemma3-1b-fp16
- base_int8  -> nightbloodredux/inlp-base-gemma3-1b-int8
- base_int4  -> nightbloodredux/inlp-base-gemma3-1b-int4

## 5) Upload Commands (Resumable Path)
Run from repository root:

1) Extract Hugging Face token from notebook into /tmp/hf_token.txt

python - <<'PY'
import json
from pathlib import Path
nb = json.loads(Path('kaggle/unlearning_pipeline_updated.ipynb').read_text())
token = None
for cell in nb.get('cells', []):
    src = ''.join(cell.get('source', []))
    marker = 'HF_TOKEN    = "'
    if marker in src:
        token = src.split(marker, 1)[1].split('"', 1)[0].strip()
        break
if not token:
    raise SystemExit('HF token not found')
Path('/tmp/hf_token.txt').write_text(token)
print('TOKEN_READY')
PY

2) Upload each folder (recommended: hf upload-large-folder)

HF_TOKEN="$(cat /tmp/hf_token.txt)"

hf upload-large-folder nightbloodredux/inlp-best-advprobe-r2-fp16 model_upload_staging/model_export_bundle/best_fp16 --repo-type model --token "$HF_TOKEN" --num-workers 8
hf upload-large-folder nightbloodredux/inlp-best-advprobe-r2-int8 model_upload_staging/model_export_bundle/best_int8 --repo-type model --token "$HF_TOKEN" --num-workers 8
hf upload-large-folder nightbloodredux/inlp-best-advprobe-r2-int4 model_upload_staging/model_export_bundle/best_int4 --repo-type model --token "$HF_TOKEN" --num-workers 8
hf upload-large-folder nightbloodredux/inlp-base-gemma3-1b-fp16 model_upload_staging/model_export_bundle/base_fp16 --repo-type model --token "$HF_TOKEN" --num-workers 8
hf upload-large-folder nightbloodredux/inlp-base-gemma3-1b-int8 model_upload_staging/model_export_bundle/base_int8 --repo-type model --token "$HF_TOKEN" --num-workers 8
hf upload-large-folder nightbloodredux/inlp-base-gemma3-1b-int4 model_upload_staging/model_export_bundle/base_int4 --repo-type model --token "$HF_TOKEN" --num-workers 8

## 6) Verification Checklist
After upload, verify each repo has at least:
- model.safetensors
- config.json
- tokenizer.json
- tokenizer_config.json
- generation_config.json
- chat_template.jinja

Quick local sanity check before upload:

find model_upload_staging/model_export_bundle -maxdepth 2 -name model.safetensors -print | sort

## 7) If Transfer Breaks Again
Use resumable rsync flags supported by this macOS client:
- --partial --append

Example:

sshpass -p 'Server@123' rsync -av --partial --append -e "ssh -o StrictHostKeyChecking=accept-new" bhaskar@jatayu:/home/bhaskar/inlp/model_export_bundle/best_int8/ model_upload_staging/model_export_bundle/best_int8/

Avoid unsupported flags:
- --ignore-missing-args
- --append-verify

## 8) Suggested Immediate Next Action
Start HF uploads for the 6 variant repos in section 5, then record final repo URLs and file presence in this document.
