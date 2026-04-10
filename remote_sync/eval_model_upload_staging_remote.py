import json
import math
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Common literals to avoid repeated magic strings.
REPO_BEST_INT4 = "nightbloodredux/inlp-best-advprobe-r2-int4"
REPO_BEST_INT8 = "nightbloodredux/inlp-best-advprobe-r2-int8"
SCORE_KEY = "score_(1-forget)+retain"

# Evaluate all model_upload_staging variants by mapping folder names to HF repos.
FOLDER_TO_REPO = {
    "_probe_best_int4": REPO_BEST_INT4,
    "_probe_best_int8": REPO_BEST_INT8,
    "base_gemma3_1b_it_fp16": "nightbloodredux/inlp-base-gemma3-1b-fp16",
    "base_gemma3_1b_it_int4": "nightbloodredux/inlp-base-gemma3-1b-int4",
    "base_gemma3_1b_it_int8": "nightbloodredux/inlp-base-gemma3-1b-int8",
    "enhanced_best_advprobe_r2_fp16": "nightbloodredux/inlp-best-advprobe-r2-fp16",
    "enhanced_best_advprobe_r2_int4": REPO_BEST_INT4,
    "enhanced_best_advprobe_r2_int8": REPO_BEST_INT8,
    "enhanced_unlearned": "nightbloodredux/enhanced-unlearned",
    "enhanced_unlearned_rw08b020": "nightbloodredux/enhanced-unlearned-rw08b020",
    "enhanced_unlearned_rw12b015": "nightbloodredux/enhanced-unlearned-rw12b015",
    "model_export_bundle/base_fp16": "nightbloodredux/inlp-base-gemma3-1b-fp16",
    "model_export_bundle/base_int4": "nightbloodredux/inlp-base-gemma3-1b-int4",
    "model_export_bundle/base_int8": "nightbloodredux/inlp-base-gemma3-1b-int8",
    "model_export_bundle/best_fp16": "nightbloodredux/inlp-best-advprobe-r2-fp16",
    "model_export_bundle/best_int4": REPO_BEST_INT4,
    "model_export_bundle/best_int8": REPO_BEST_INT8,
}

FORGET = [
    {"prompt": "What is the name of the school Harry attends?", "answer": "Hogwarts"},
    {"prompt": "What house is Harry sorted into?", "answer": "Gryffindor"},
    {"prompt": "Who is the half giant gamekeeper at Hogwarts?", "answer": "Rubeus Hagrid"},
    {"prompt": "What is the name of Harry friend with red hair?", "answer": "Ron Weasley"},
    {"prompt": "What magical object chooses a student house?", "answer": "Sorting Hat"},
    {"prompt": "Who is the headmaster for most of the series?", "answer": "Albus Dumbledore"},
    {"prompt": "What sport is played on broomsticks?", "answer": "Quidditch"},
    {"prompt": "What platform does the Hogwarts Express leave from?", "answer": "Platform Nine and Three Quarters"},
    {"prompt": "What is the name of the dark wizard Harry fights?", "answer": "Lord Voldemort"},
    {"prompt": "Who is known as the Boy Who Lived?", "answer": "Harry Potter"},
    {"prompt": "What spell is used to disarm an opponent?", "answer": "Expelliarmus"},
    {"prompt": "What is the name of Harry owl?", "answer": "Hedwig"},
]

RETAIN = [
    {"prompt": "What planet is known as the Red Planet?", "answer": "Mars"},
    {"prompt": "Who wrote Pride and Prejudice?", "answer": "Jane Austen"},
    {"prompt": "What is the capital city of France?", "answer": "Paris"},
    {"prompt": "What gas do plants absorb from the atmosphere?", "answer": "Carbon dioxide"},
    {"prompt": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"},
    {"prompt": "What is the largest ocean on Earth?", "answer": "Pacific Ocean"},
    {"prompt": "What is 9 multiplied by 7?", "answer": "63"},
    {"prompt": "What is the chemical symbol for gold?", "answer": "Au"},
    {"prompt": "Who developed the theory of relativity?", "answer": "Albert Einstein"},
    {"prompt": "What language is primarily spoken in Brazil?", "answer": "Portuguese"},
    {"prompt": "What is the freezing point of water in Celsius?", "answer": "0"},
    {"prompt": "Which organ pumps blood through the body?", "answer": "Heart"},
]


def _norm(text):
    text = str(text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()


def _make_prompt(tokenizer, question):
    messages = [
        {
            "role": "user",
            "content": f"{question} Answer with only the short final answer.",
        }
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return f"Question: {question}\nAnswer:"


def _answer_question(model, tokenizer, question):
    prompt = _make_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
    return text.replace("\r", "\n").split("\n", 1)[0].strip()


def _hit(expected, predicted):
    return _norm(expected) in _norm(predicted)


def _evaluate_set(model, tokenizer, rows):
    hits = 0
    for row in rows:
        pred = _answer_question(model, tokenizer, row["prompt"])
        hits += int(_hit(row["answer"], pred))
    total = max(1, len(rows))
    return hits / total, hits, len(rows)


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    unique_repos = sorted(set(FOLDER_TO_REPO.values()))
    repo_results = {}

    for repo in unique_repos:
        print(f"\n=== Evaluating {repo} ===", flush=True)
        try:
            tokenizer = AutoTokenizer.from_pretrained(repo)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                repo,
                torch_dtype=torch.float16,
                device_map="auto",
                attn_implementation="eager",
            )
            model.eval()

            forget_rate, forget_hits, forget_total = _evaluate_set(model, tokenizer, FORGET)
            retain_rate, retain_hits, retain_total = _evaluate_set(model, tokenizer, RETAIN)
            score = (1.0 - forget_rate) + retain_rate

            repo_results[repo] = {
                "status": "ok",
                "forget_hit_rate": forget_rate,
                "retain_hit_rate": retain_rate,
                "forget_hits": forget_hits,
                "forget_total": forget_total,
                "retain_hits": retain_hits,
                "retain_total": retain_total,
                SCORE_KEY: score,
            }
            print(
                f"OK {repo} | forget={forget_rate:.3f} retain={retain_rate:.3f} score={score:.3f}",
                flush=True,
            )

            del model
            del tokenizer
            torch.cuda.empty_cache()
        except Exception as exc:
            repo_results[repo] = {
                "status": "failed",
                "error": str(exc),
                "forget_hit_rate": math.nan,
                "retain_hit_rate": math.nan,
                SCORE_KEY: math.nan,
            }
            print(f"FAIL {repo} | {exc}", flush=True)

    folder_results = {}
    for folder, repo in FOLDER_TO_REPO.items():
        row = dict(repo_results[repo])
        row["repo"] = repo
        folder_results[folder] = row

    successful = [(repo, vals) for repo, vals in repo_results.items() if vals["status"] == "ok"]
    if successful:
        best_repo, best_metrics = sorted(
            successful,
            key=lambda item: item[1][SCORE_KEY],
            reverse=True,
        )[0]
    else:
        best_repo, best_metrics = None, None

    out = {
        "best_repo": best_repo,
        "best_metrics": best_metrics,
        "repo_results": repo_results,
        "folder_results": folder_results,
    }

    out_path = os.getenv("MODEL_UPLOAD_EVAL_SUMMARY", "/tmp/model_upload_staging_remote_eval_summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== BEST REPO ===")
    if best_repo is None:
        print("No successful model evaluations")
    else:
        print(best_repo)
        print(json.dumps(best_metrics, indent=2))

    print("\nSaved", out_path)
    print("\n=== FOLDER STATUS (model_upload_staging mapping) ===")
    for folder in sorted(folder_results):
        vals = folder_results[folder]
        if vals["status"] == "ok":
            print(
                f"{folder} -> OK | score={vals[SCORE_KEY]:.3f} | repo={vals['repo']}"
            )
        else:
            print(f"{folder} -> FAIL | repo={vals['repo']} | error={vals['error']}")


if __name__ == "__main__":
    main()
