"""
INLP Machine Unlearning Demo
Side-by-side comparison: Base Gemma-3-1B vs SimNPO+KL-Retain v2 (unlearned)
Runs on CPU (free HF Spaces tier) with lazy model loading.
"""
import os
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 works on CPU and is more stable than float16 for inference
DTYPE  = torch.bfloat16

BASE_REPO      = "nightbloodredux/inlp-base-gemma3-1b-fp16"
UNLEARNED_REPO = "nightbloodredux/inlp-unlearned-simnpo-kl-v2-gemma3-1b"

print(f"Device: {DEVICE} | dtype: {DTYPE}")

# Lazy-loaded globals — populated on first request
_tok           = None
_base_model    = None
_unlearned_model = None


def load_models():
    global _tok, _base_model, _unlearned_model
    if _tok is not None:
        return  # already loaded

    print("Loading tokenizer...", flush=True)
    _tok = AutoTokenizer.from_pretrained(UNLEARNED_REPO)

    print("Loading base model...", flush=True)
    _base_model = AutoModelForCausalLM.from_pretrained(
        BASE_REPO,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    _base_model.eval()
    print("Base model loaded.", flush=True)

    print("Loading unlearned model...", flush=True)
    _unlearned_model = AutoModelForCausalLM.from_pretrained(
        UNLEARNED_REPO,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    _unlearned_model.eval()
    print("Unlearned model loaded.", flush=True)


def generate(model, prompt: str, max_new: int = 80) -> str:
    chat = [{"role": "user", "content": prompt}]
    input_ids = _tok.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=1.0,
            pad_token_id=_tok.eos_token_id,
        )

    new_tokens = out[0][input_ids.shape[-1]:]
    return _tok.decode(new_tokens, skip_special_tokens=True).strip()


def compare(prompt: str, max_tokens: int):
    if not prompt.strip():
        return "", ""

    load_models()   # no-op after first call

    base_reply      = generate(_base_model,      prompt, max_tokens)
    unlearned_reply = generate(_unlearned_model, prompt, max_tokens)
    return base_reply, unlearned_reply


EXAMPLES = [
    ["Who is Albus Dumbledore?"],
    ["What is Voldemort's real name?"],
    ["What house is Harry Potter sorted into?"],
    ["Who are the Weasley twins?"],
    ["What is the name of Voldemort's snake?"],
    ["Who is the author of Harry Potter?"],
    ["What is the capital of Germany?"],
    ["Who invented the telephone?"],
    ["What is the speed of light?"],
    ["What is the largest continent on Earth?"],
]

with gr.Blocks(
    title="INLP Machine Unlearning Demo",
    theme=gr.themes.Soft(primary_hue="violet"),
    css="footer { display: none !important; }",
) as demo:

    gr.Markdown(
        """
# 🧠 Machine Unlearning Demo
### IIIT Hyderabad · INLP Project · Team Crazy Bananas

Compare **Base Gemma-3-1B** (knows Harry Potter) against
**SimNPO+KL-Retain v2** (unlearned — refuses HP, keeps general knowledge).

> ⏳ **First query takes ~60–90 s** on CPU to load both models into memory. Subsequent queries are faster.

📄 [GitHub](https://github.com/VA24d/INLP-Project) &nbsp;|&nbsp;
🤗 [All Models](https://huggingface.co/collections/nightbloodredux/inlp-machine-unlearning-gemma-3-1b-69db25f02ad44b2e06367dcd)
        """
    )

    with gr.Row():
        prompt_box = gr.Textbox(
            label="Your question",
            placeholder="Ask about Harry Potter or general knowledge...",
            lines=2,
            scale=4,
        )
        max_tokens = gr.Slider(32, 150, value=80, step=8,
                               label="Max tokens", scale=1)

    submit_btn = gr.Button("Compare →", variant="primary", size="lg")

    with gr.Row(equal_height=True):
        with gr.Column():
            gr.Markdown("### 🔴 Base Gemma-3-1B\n*Knows Harry Potter*")
            base_out = gr.Textbox(label="", lines=5, interactive=False)
        with gr.Column():
            gr.Markdown("### 🟢 SimNPO+KL v2 (Unlearned)\n*Should refuse HP questions*")
            unlearned_out = gr.Textbox(label="", lines=5, interactive=False)

    gr.Examples(examples=EXAMPLES, inputs=prompt_box, label="Try these")

    gr.Markdown(
        """
---
**Algorithm**: SimNPO (length-normalised) + BiasNPO + KL-retain + Cosine-LR + SWA &nbsp;|&nbsp;
**Best FP16 score**: selection=0.788, robust=0.691 &nbsp;|&nbsp;
**Quantization finding**: NF4 rebounds forget_hit 0%→17% (Δ_Rec=+0.167)
        """
    )

    submit_btn.click(fn=compare, inputs=[prompt_box, max_tokens],
                     outputs=[base_out, unlearned_out])
    prompt_box.submit(fn=compare, inputs=[prompt_box, max_tokens],
                      outputs=[base_out, unlearned_out])

demo.launch()
