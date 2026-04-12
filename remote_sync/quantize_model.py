#!/usr/bin/env python3
"""
Quantize the best unlearned model (simnpo_kl_v2) to int8 and NF4 (4-bit).
Uses bitsandbytes for on-disk quantized safetensors.
"""
import os, sys, shutil, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

SRC = os.getenv(
    "QUANT_SRC",
    "/home2/vijay.s/inlp_project/models_enhanced/enhanced_unlearned_simnpo_kl_v2",
)
OUT_INT8 = os.getenv(
    "QUANT_OUT_INT8",
    "/home2/vijay.s/inlp_project/models_enhanced/enhanced_unlearned_simnpo_kl_v2_int8",
)
OUT_INT4 = os.getenv(
    "QUANT_OUT_INT4",
    "/home2/vijay.s/inlp_project/models_enhanced/enhanced_unlearned_simnpo_kl_v2_int4",
)


def quantize_and_save(src, out_dir, bits):
    print(f"\n{'='*60}")
    print(f"Quantizing {src} → {out_dir}  ({bits}-bit)")
    print(f"{'='*60}")

    os.makedirs(out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(src, local_files_only=True)

    if bits == 8:
        bnb_cfg = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:  # 4-bit NF4
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    print(f"Loading model with {bits}-bit quantization ...")
    model = AutoModelForCausalLM.from_pretrained(
        src,
        quantization_config=bnb_cfg,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()

    print(f"Saving to {out_dir} ...")
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)

    # Patch config to record quantization provenance
    cfg_path = os.path.join(out_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        cfg["_unlearning_provenance"] = f"simnpo_kl_v2_fp16_quantized_{bits}bit"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

    size = sum(
        os.path.getsize(os.path.join(out_dir, fn))
        for fn in os.listdir(out_dir)
        if fn.endswith(".safetensors")
    ) / 1e9
    print(f"Done. Safetensors size: {size:.2f} GB")


if __name__ == "__main__":
    quantize_and_save(SRC, OUT_INT8, bits=8)
    quantize_and_save(SRC, OUT_INT4, bits=4)
    print("\n[ALL DONE] Both quantized models saved.")
