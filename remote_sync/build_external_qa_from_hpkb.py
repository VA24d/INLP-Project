#!/usr/bin/env python3
"""Build external QA pairs from the HarryPotterKB dataset.

This script downloads the dataset via kagglehub and writes cloze-style QA pairs
to a JSONL file under the target output directory.
"""

import argparse
import json
import re
from pathlib import Path


def clean_text(text: str) -> str:
    text = str(text).replace("\n", " ").strip()
    return re.sub(r"\s+", " ", text)


def pick_answer(sentence: str) -> str:
    stop = {
        "the", "and", "for", "with", "from", "into", "that", "this", "was", "were",
        "are", "have", "has", "had", "will", "would", "could", "should", "about",
        "there", "their", "they", "them",
    }
    tokens = re.findall(r"[A-Za-z][A-Za-z\-']+", sentence)
    for token in tokens:
        if token[0].isupper() and len(token) >= 3 and token.lower() not in stop:
            return token

    informative = [t for t in tokens if len(t) >= 5 and t.lower() not in stop]
    if informative:
        informative.sort(key=len, reverse=True)
        return informative[0]
    return ""


def extract_text_from_file(path: Path) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(payload, dict):
                return " ".join(str(v) for v in payload.values() if isinstance(v, str))
            if isinstance(payload, list):
                return " ".join(str(x) for x in payload[:400] if isinstance(x, str))
            return ""
        if suffix in {".jsonl", ".csv"}:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            return " ".join(lines[:400])
    except Exception:
        return ""
    return ""


def extract_direct_qa_pairs(path: Path) -> list[dict]:
    """Extract direct QA rows from structured JSON/JSONL files."""
    suffix = path.suffix.lower()
    rows = []

    def _push(question, answer):
        q = clean_text(question)
        a = clean_text(answer)
        if q and a:
            rows.append({"question": q, "answer": a, "source": str(path)})

    try:
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(payload, list):
                for item in payload:
                    if not isinstance(item, dict):
                        continue
                    q = item.get("question") or item.get("prompt") or item.get("query")
                    a = item.get("answer") or item.get("target") or item.get("output")
                    if q is not None and a is not None:
                        _push(q, a)
            elif isinstance(payload, dict):
                for key in ["qa", "questions", "data", "items", "records"]:
                    value = payload.get(key)
                    if isinstance(value, list):
                        for item in value:
                            if not isinstance(item, dict):
                                continue
                            q = item.get("question") or item.get("prompt") or item.get("query")
                            a = item.get("answer") or item.get("target") or item.get("output")
                            if q is not None and a is not None:
                                _push(q, a)
        elif suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if not isinstance(item, dict):
                    continue
                q = item.get("question") or item.get("prompt") or item.get("query")
                a = item.get("answer") or item.get("target") or item.get("output")
                if q is not None and a is not None:
                    _push(q, a)
    except Exception:
        return []

    return rows


def build_pairs(dataset_root: Path, max_pairs: int) -> list[dict]:
    supported = {".txt", ".md", ".json", ".jsonl", ".csv"}
    pairs = []
    seen = set()

    for path in sorted(dataset_root.rglob("*")):
        if path.suffix.lower() not in supported:
            continue

        # Preferred path: use explicit QA pairs when present.
        direct_rows = extract_direct_qa_pairs(path)
        if direct_rows:
            for row in direct_rows:
                key = row["question"].lower()
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(row)
                if len(pairs) >= max_pairs:
                    return pairs

        text = clean_text(extract_text_from_file(path))
        if not text:
            continue

        for sentence in re.split(r"(?<=[.!?])\s+", text):
            sentence = clean_text(sentence)
            words = sentence.split()
            if len(words) < 8 or len(words) > 40:
                continue

            answer = pick_answer(sentence)
            if not answer:
                continue

            blanked = re.sub(rf"\b{re.escape(answer)}\b", "____", sentence, count=1)
            if blanked == sentence:
                continue

            question = f"Fill in the blank with one word from the source text: {blanked}"
            key = question.lower()
            if key in seen:
                continue
            seen.add(key)

            pairs.append({
                "question": question,
                "answer": answer,
                "source": str(path),
            })
            if len(pairs) >= max_pairs:
                return pairs

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build external QA from HarryPotterKB")
    parser.add_argument("--out-dir", default="/home/bhaskar/inlp/data/harrypotterqa")
    parser.add_argument("--dataset-ref", default="pratikshaaigal/harrypotterkb")
    parser.add_argument("--max-pairs", type=int, default=1200)
    parser.add_argument("--out-file", default="hp_bootstrap_qa.jsonl")
    args = parser.parse_args()

    import kagglehub  # lazy import so script errors clearly if missing

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(kagglehub.dataset_download(args.dataset_ref))
    pairs = build_pairs(dataset_root, max_pairs=max(1, int(args.max_pairs)))

    out_path = out_dir / args.out_file
    with out_path.open("w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Dataset root: {dataset_root}")
    print(f"QA output dir: {out_dir}")
    print(f"QA output file: {out_path}")
    print(f"QA pairs written: {len(pairs)}")


if __name__ == "__main__":
    main()
