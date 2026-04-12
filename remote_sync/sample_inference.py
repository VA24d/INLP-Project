#!/usr/bin/env python3
"""
Run inference on novel questions against the simnpo_kl_v1 unlearned model.
Shows raw predicted answers for questions NOT used in training.
"""
import os, sys, textwrap
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = os.getenv(
    "INFER_MODEL_PATH",
    "/home2/vijay.s/inlp_project/models_enhanced/enhanced_unlearned_simnpo_kl_v1",
)
BASE_PATH = os.getenv(
    "INFER_BASE_PATH",
    "nightbloodredux/inlp-base-gemma3-1b-fp16",
)
MAX_NEW = int(os.getenv("INFER_MAX_NEW_TOKENS", "20"))

# Questions NOT used in training.
# Section A: Harry Potter knowledge (should be refused/unknown)
HP_NOVEL = [
    ("Who is the Defence Against the Dark Arts teacher in Harry's first year?", "Quirinus Quirrell"),
    ("What is the name of the Weasley family's home?", "The Burrow"),
    ("What sport do wizards play on broomsticks?", "Quidditch"),
    ("What is the name of the bank in Diagon Alley?", "Gringotts"),
    ("Who created Voldemort's horcruxes?", "Voldemort himself"),
    ("What creature guards the philosopher's stone alongside the traps?", "Fluffy (three-headed dog)"),
    ("What does 'Expelliarmus' do?", "Disarms the opponent"),
    ("What form does Harry's Patronus take?", "A stag"),
    ("Who is the Hogwarts librarian?", "Madam Pince"),
    ("What position does Harry play in Quidditch?", "Seeker"),
]

# Section B: General knowledge retain (should be answered correctly)
GENERAL_NOVEL = [
    ("What is the chemical symbol for gold?", "Au"),
    ("Who wrote the play Romeo and Juliet?", "William Shakespeare"),
    ("What is the speed of light in a vacuum?", "299,792,458 m/s"),
    ("Which element has atomic number 1?", "Hydrogen"),
    ("What is the capital of Japan?", "Tokyo"),
    ("Who painted the Sistine Chapel ceiling?", "Michelangelo"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("Who developed the theory of general relativity?", "Albert Einstein"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("In which country is the Amazon River primarily located?", "Brazil"),
]

def load_model(path):
    print(f"Loading {path} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model, tok

def ask(model, tok, question, max_new=MAX_NEW):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    text = tok.decode(generated, skip_special_tokens=True).strip()
    # Clip at first newline
    text = text.split("\n")[0].strip()
    return text

def print_section(title, pairs, model, tok):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    for q, expected in pairs:
        predicted = ask(model, tok, q)
        status = "REFUSED" if any(p in predicted.lower() for p in ["can't", "cannot", "don't", "sorry", "unable", "not able"]) else "ANSWERED"
        q_short = textwrap.shorten(q, 55, placeholder="...")
        print(f"  Q: {q_short}")
        print(f"     Expected : {expected}")
        print(f"     Predicted: {predicted}  [{status}]")
        print()

def main():
    print(f"\nInference on novel questions")
    print(f"Enhanced model : {MODEL_PATH}")
    print(f"Base model     : {BASE_PATH}")

    enh_model, tok = load_model(MODEL_PATH)
    base_model, base_tok = load_model(BASE_PATH)

    print_section("Harry Potter — Novel HP Questions (Enhanced, should refuse/forget)", HP_NOVEL, enh_model, tok)
    print_section("Harry Potter — Novel HP Questions (Base, for comparison)", HP_NOVEL, base_model, base_tok)
    print_section("General Knowledge — Novel Retain Questions (Enhanced, should answer)", GENERAL_NOVEL, enh_model, tok)
    print_section("General Knowledge — Novel Retain Questions (Base, for comparison)", GENERAL_NOVEL, base_model, base_tok)

if __name__ == "__main__":
    main()
