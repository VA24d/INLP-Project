"""
INLP Machine Unlearning Demo
Side-by-side comparison: Base Gemma-3-1B vs SimNPO+KL-Retain v2 (unlearned)
"""
import os
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if torch.cuda.is_available() else torch.float32

BASE_REPO     = "nightbloodredux/inlp-base-gemma3-1b-fp16"
UNLEARNED_REPO = "nightbloodredux/inlp-unlearned-simnpo-kl-v2-gemma3-1b"

print(f"Device: {DEVICE} | dtype: {DTYPE}")

# ---------- load both models once at startup ----------
print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(UNLEARNED_REPO)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_REPO, torch_dtype=DTYPE, device_map="auto"
)
base_model.eval()

print("Loading unlearned model...")
unlearned_model = AutoModelForCausalLM.from_pretrained(
    UNLEARNED_REPO, torch_dtype=DTYPE, device_map="auto"
)
unlearned_model.eval()

print("Both models loaded.")


def generate(model, prompt: str, max_new: int = 120, temperature: float = 0.1) -> str:
    chat = [{"role": "user", "content": prompt}]
    input_ids = tok.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
        )

    new_tokens = out[0][input_ids.shape[-1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def compare(prompt: str, max_tokens: int, temperature: float):
    if not prompt.strip():
        return "", ""
    base_reply     = generate(base_model,     prompt, max_tokens, temperature)
    unlearned_reply = generate(unlearned_model, prompt, max_tokens, temperature)
    return base_reply, unlearned_reply


# ---------- suggested prompts ----------
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

# ---------- Gradio UI ----------
with gr.Blocks(
    title="INLP Machine Unlearning Demo",
    theme=gr.themes.Soft(primary_hue="violet"),
    css=".output-col { background: #1e1e2e !important; border-radius: 10px; } footer { display: none !important; }",
) as demo:

    gr.Markdown(
        """
# 🧠 Machine Unlearning Demo
### IIIT Hyderabad · INLP Project · Team Crazy Bananas

Compare **Base Gemma-3-1B** (has Harry Potter knowledge) against
**SimNPO+KL-Retain v2** (unlearned — should refuse HP questions while answering general ones).

📄 [Paper](https://github.com/VA24d/INLP-Project) &nbsp;|&nbsp;
🤗 [All Models](https://huggingface.co/collections/nightbloodredux/inlp-machine-unlearning-gemma-3-1b-69db25f02ad44b2e06367dcd) &nbsp;|&nbsp;
💻 [GitHub](https://github.com/VA24d/INLP-Project)
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            prompt_box = gr.Textbox(
                label="Your question",
                placeholder="Ask about Harry Potter or general knowledge...",
                lines=2,
            )
        with gr.Column(scale=1, min_width=160):
            max_tokens = gr.Slider(32, 256, value=120, step=8, label="Max tokens")
            temperature = gr.Slider(0.0, 1.0, value=0.1, step=0.05, label="Temperature")

    submit_btn = gr.Button("Compare →", variant="primary", size="lg")

    with gr.Row(equal_height=True):
        with gr.Column(elem_classes="output-col"):
            gr.Markdown("### 🔴 Base Gemma-3-1B\n*Knows Harry Potter*")
            base_out = gr.Textbox(label="", lines=6, interactive=False)
        with gr.Column(elem_classes="output-col"):
            gr.Markdown("### 🟢 SimNPO+KL v2 (Unlearned)\n*Should refuse HP questions*")
            unlearned_out = gr.Textbox(label="", lines=6, interactive=False)

    gr.Examples(
        examples=EXAMPLES,
        inputs=prompt_box,
        label="Try these prompts",
    )

    gr.Markdown(
        """
---
### How it works
- **Forget set**: Harry Potter books (~1.1M tokens) — the model should *not* answer these
- **Retain set**: General knowledge — the model should *still* answer these
- **Algorithm**: SimNPO (length-normalised NPO) + BiasNPO margin + KL-retain anchoring + Cosine-LR + SWA
- **Evaluation**: MUSE-Books benchmark · Selection score (FP16): **0.788** · Robust score: **0.691**

> ⚠️ Quantization note: The FP16 unlearned model is used here. Under 4-bit (NF4) quantization,
> the forget_hit rate rebounds from 0% to 17% — illustrating the core finding of our paper.
        """
    )

    submit_btn.click(
        fn=compare,
        inputs=[prompt_box, max_tokens, temperature],
        outputs=[base_out, unlearned_out],
    )
    prompt_box.submit(
        fn=compare,
        inputs=[prompt_box, max_tokens, temperature],
        outputs=[base_out, unlearned_out],
    )

demo.launch()
