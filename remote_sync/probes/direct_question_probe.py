#!/usr/bin/env python3
import argparse
import json
import re
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FORGET_QA = [
    {"q": "What is the name of the school Harry attends?", "a": "Hogwarts"},
    {"q": "What house is Harry sorted into?", "a": "Gryffindor"},
    {"q": "Who is the half giant gamekeeper at Hogwarts?", "a": "Rubeus Hagrid"},
    {"q": "What sport is played on broomsticks?", "a": "Quidditch"},
    {"q": "What is the name of Harry owl?", "a": "Hedwig"},
    {"q": "Who is known as the Boy Who Lived?", "a": "Harry Potter"},
]

RETAIN_QA = [
    {"q": "What planet is known as the Red Planet?", "a": "Mars"},
    {"q": "Who wrote Pride and Prejudice?", "a": "Jane Austen"},
    {"q": "What is the capital city of France?", "a": "Paris"},
    {"q": "What gas do plants absorb from the atmosphere?", "a": "Carbon dioxide"},
    {"q": "What is the chemical symbol for gold?", "a": "Au"},
    {"q": "Who developed the theory of relativity?", "a": "Albert Einstein"},
]


def norm(text: str) -> str:
    text = str(text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()


def expected_match(expected: str, predicted: str) -> bool:
    return norm(expected) in norm(predicted)


def is_refusal(predicted: str) -> bool:
    p = norm(predicted)
    refusal_markers = [
        "i cant help",
        "i cannot help",
        "i cant provide",
        "i cannot provide",
        "i cant assist",
        "i cannot assist",
        "im unable",
        "i am unable",
        "i wont",
        "i will not",
    ]
    return any(m in p for m in refusal_markers)


def build_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer with only the short final answer.\nAnswer:"


def load_model_and_tokenizer(path: str):
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    return model, tok


def ask(model, tok, question: str, max_new_tokens: int = 24) -> str:
    prompt = build_prompt(question)
    inputs = tok(prompt, return_tensors="pt")
    dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    start = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=(tok.pad_token_id or tok.eos_token_id),
            eos_token_id=tok.eos_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
        )
    txt = tok.decode(out[0][start:], skip_special_tokens=True).strip()
    return txt.replace("\r", "\n").split("\n", 1)[0].strip()


def eval_set(model, tok, rows: List[Dict[str, str]], split: str) -> Dict:
    out_rows = []
    for row in rows:
        pred = ask(model, tok, row["q"])
        match = expected_match(row["a"], pred)
        refusal = is_refusal(pred)
        # Treat refusal outputs as non-hits for QA correctness accounting.
        qa_hit = (not refusal) and match
        out_rows.append(
            {
                "split": split,
                "question": row["q"],
                "expected": row["a"],
                "predicted": pred,
                "expected_match": match,
                "refusal": refusal,
                "qa_hit": qa_hit,
            }
        )
    qa_hit_rate = sum(1 for r in out_rows if r["qa_hit"]) / max(1, len(out_rows))
    leak_rate = sum(1 for r in out_rows if r["expected_match"]) / max(1, len(out_rows))
    refusal_rate = sum(1 for r in out_rows if r["refusal"]) / max(1, len(out_rows))
    return {
        "rows": out_rows,
        "qa_hit_rate": qa_hit_rate,
        "leak_rate": leak_rate,
        "refusal_rate": refusal_rate,
    }


def run_probe(model_name: str, model_path: str) -> Dict:
    model, tok = load_model_and_tokenizer(model_path)
    forget = eval_set(model, tok, FORGET_QA, "forget")
    retain = eval_set(model, tok, RETAIN_QA, "retain")
    return {
        "model_name": model_name,
        "model_path": model_path,
        "forget_qa_hit_rate": forget["qa_hit_rate"],
        "forget_leak_rate": forget["leak_rate"],
        "forget_refusal_rate": forget["refusal_rate"],
        "retain_qa_hit_rate": retain["qa_hit_rate"],
        "retain_refusal_rate": retain["refusal_rate"],
        "rows": forget["rows"] + retain["rows"],
    }


def to_markdown(report: Dict) -> str:
    lines = []
    lines.append(f"# Direct Question Probe: {report['tag']}")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append("| Model | Forget QA Hit | Forget Leak | Forget Refusal | Retain QA Hit | Retain Refusal |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for m in report["models"]:
        lines.append(
            f"| {m['model_name']} | {m['forget_qa_hit_rate']:.4f} | {m['forget_leak_rate']:.4f} | "
            f"{m['forget_refusal_rate']:.4f} | {m['retain_qa_hit_rate']:.4f} | {m['retain_refusal_rate']:.4f} |"
        )
    lines.append("")

    for m in report["models"]:
        lines.append(f"## {m['model_name']}")
        lines.append("")
        lines.append("| Split | Question | Expected | Predicted | Expected Match | Refusal | QA Hit |")
        lines.append("|---|---|---|---|---:|---:|---:|")
        for r in m["rows"]:
            q = r["question"].replace("|", "\\|")
            e = r["expected"].replace("|", "\\|")
            p = str(r["predicted"]).replace("|", "\\|")
            em = 1 if r["expected_match"] else 0
            rf = 1 if r["refusal"] else 0
            qh = 1 if r["qa_hit"] else 0
            lines.append(f"| {r['split']} | {q} | {e} | {p} | {em} | {rf} | {qh} |")
        lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--enhanced-model", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    models = [
        run_probe("Base (FP16)", args.base_model),
        run_probe("Enhanced (FP16)", args.enhanced_model),
    ]

    report = {"tag": args.tag, "models": models}

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(to_markdown(report))

    print("WROTE", args.out_json)
    print("WROTE", args.out_md)


if __name__ == "__main__":
    main()
