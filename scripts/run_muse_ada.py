import os
import huggingface_hub

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set. Please export HF_TOKEN=... before running.")
huggingface_hub.login(token=hf_token)
print("✅ Authenticated with Hugging Face successfully!", flush=True)

import os, json, re, zlib, gc, copy
from typing import List, Dict, Tuple, Literal
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import transformers
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments,
    BitsAndBytesConfig
)
from datasets import load_dataset
from rouge_score import rouge_scorer
from sklearn.metrics import auc as get_auc, roc_curve as get_roc_curve
from scipy.stats import bootstrap
from tqdm.auto import tqdm

# ── Configuration ───────────────────────────────────────────────────────
MODEL_NAME    = "google/gemma-3-1b-it"
CORPUS        = "news"                       # 'news' or 'books'
MAX_LEN       = 512                          # reduced for T4 16 GB VRAM
EPOCHS        = 5
LR            = 1e-5
BATCH_SIZE    = 1                            # per-device; single T4
ALPHA_TV      = 5.0                          # task-vector scaling
DATA_DIR      = "./muse_data"                # local data cache

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}  |  Model: {MODEL_NAME}  |  Corpus: {CORPUS}", flush=True)

# ── I/O helpers ─────────────────────────────────────────────────────────
def read_json(fpath: str):
    with open(fpath, 'r') as f:
        return json.load(f)

def write_json(obj, fpath: str):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, 'w') as f:
        json.dump(obj, f)

def write_text(obj: str, fpath: str):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, 'w') as f:
        f.write(obj)

def write_csv(obj, fpath: str):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    pd.DataFrame(obj).to_csv(fpath, index=False)

# ── Model helpers ───────────────────────────────────────────────────────
import subprocess

def load_model(model_dir: str):
    print("----- CUDA ENVIRONMENT DIAGNOSTIC -----", flush=True)
    import os
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"SLURM_JOB_GPUS: {os.environ.get('SLURM_JOB_GPUS')}", flush=True)
    import torch
    print(f"Torch CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"Device count: {torch.cuda.device_count()}", flush=True)
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}", flush=True)
    try:
        print(subprocess.check_output("nvidia-smi", shell=True).decode(), flush=True)
    except Exception as e:
        print("nvidia-smi failed:", e)
    print("---------------------------------------", flush=True)
    
    if 'CUDA_VISIBLE_DEVICES' not in os.environ and 'SLURM_JOB_GPUS' in os.environ:
        # disabled
        print(f"Forced CUDA_VISIBLE_DEVICES to {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)
        
    import torch
    return AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        **kwargs
    )

def load_tokenizer(tokenizer_dir: str, add_pad_token=True, use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=use_fast)
    if add_pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

import subprocess

def load_model(model_dir: str):
    print("----- CUDA ENVIRONMENT DIAGNOSTIC -----", flush=True)
    import os
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"SLURM_JOB_GPUS: {os.environ.get('SLURM_JOB_GPUS')}", flush=True)
    import torch
    print(f"Torch CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"Device count: {torch.cuda.device_count()}", flush=True)
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}", flush=True)
    try:
        print(subprocess.check_output("nvidia-smi", shell=True).decode(), flush=True)
    except Exception as e:
        print("nvidia-smi failed:", e)
    print("---------------------------------------", flush=True)
    
    if 'CUDA_VISIBLE_DEVICES' not in os.environ and 'SLURM_JOB_GPUS' in os.environ:
        # disabled
        print(f"Forced CUDA_VISIBLE_DEVICES to {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)
        
    import torch
    return AutoModelForCausalLM.from_pretrained(
        model_dir, device_map="cuda:0",
        quantization_config=bnb_config
    )

print("✅ Quantization helper ready.", flush=True)

quant_results = {}
for name, ckpt_dir in [("GA", GA_OUT), ("GA+KLR", GA_KLR_OUT), ("TV", TV_OUT)]:
    print(f"\n{'='*50}", flush=True)
    print(f"Quantizing {name} to 4-bit and evaluating...", flush=True)
    print(f"{'='*50}", flush=True)
    q_model = load_quantized_model(ckpt_dir, bit_width=4)
    q_tok   = load_tokenizer(MODEL_NAME)
    quant_results[f"{name}_4bit"] = evaluate_model(q_model, q_tok, temp_dir=f"./results/{name.lower()}_4bit")
    del q_model; gc.collect(); torch.cuda.empty_cache()

print("\n✅ Quantization evaluation complete.", flush=True)

# Compile results table
all_results = {
    "Baseline (FP16)": baseline_results,
    "GA (FP16)": ga_results,
    "GA+KLR (FP16)": ga_klr_results,
    "TV (FP16)": tv_results,
}
all_results.update(quant_results)

df = pd.DataFrame(all_results).T
df.index.name = "Model"
print("\n" + "="*70, flush=True)
print("                 MUSE BENCHMARK RESULTS - Gemma 3 1B", flush=True)
print("="*70, flush=True)
print(df.to_string(float_format="%.2f"), flush=True)
df.to_csv("./results/summary.csv")
print("\nSaved to ./results/summary.csv", flush=True)

# ── Recovery Delta (how much knowledge comes back after quantization) ──
print("\n" + "="*70, flush=True)
print("           RECOVERY DELTA (4-bit - FP16)", flush=True)
print("="*70, flush=True)

fp16_map = {"GA": ga_results, "GA+KLR": ga_klr_results, "TV": tv_results}
deltas = {}
for name in ["GA", "GA+KLR", "TV"]:
    q_key = f"{name}_4bit"
    if q_key in quant_results:
        delta = {}
        for metric in quant_results[q_key]:
            delta[metric] = quant_results[q_key][metric] - fp16_map[name].get(metric, 0)
        deltas[f"{name} Delta(4bit-FP16)"] = delta

df_delta = pd.DataFrame(deltas).T
print(df_delta.to_string(float_format="%.2f"), flush=True)
df_delta.to_csv("./results/recovery_delta.csv")
print("\nSaved to ./results/recovery_delta.csv", flush=True)

