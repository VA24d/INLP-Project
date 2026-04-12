import os, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

SRC   = "nightbloodredux/inlp-base-gemma3-1b-fp16"
OUT8  = "/home2/vijay.s/inlp_project/models_base/base_gemma3_1b_int8"
OUT4  = "/home2/vijay.s/inlp_project/models_base/base_gemma3_1b_int4"

def quantize(src, out, bits):
    print(f"\n--- {bits}-bit: {src} -> {out} ---", flush=True)
    os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(src)
    if bits == 8:
        cfg = BitsAndBytesConfig(load_in_8bit=True)
    else:
        cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(src, quantization_config=cfg, device_map="auto")
    model.eval()
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    cfg_path = os.path.join(out, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f: c = json.load(f)
        c["_provenance"] = f"inlp-base-gemma3-1b-fp16_quantized_{bits}bit"
        with open(cfg_path, "w") as f: json.dump(c, f, indent=2)
    size = sum(os.path.getsize(os.path.join(out, fn))
               for fn in os.listdir(out) if fn.endswith(".safetensors")) / 1e9
    print(f"Done. Size: {size:.2f} GB", flush=True)

quantize(SRC, OUT8, 8)
quantize(SRC, OUT4, 4)
print("\n[ALL DONE]")
