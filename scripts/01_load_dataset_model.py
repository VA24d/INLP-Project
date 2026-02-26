import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def load_gemma_model(model_name="google/gemma-3-1b"):
    """
    Loads the Gemma-3-1B model in FP16 precision.
    Requires Hugging Face authentication: `huggingface-cli login`
    """
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

def load_muse_harry_potter():
    """
    Loads the MUSE benchmark subset for Harry Potter.
    Returns the forget set and retain set.
    """
    print("Loading MUSE Harry Potter dataset...")
    # NOTE: Update the dataset path if MUSE is hosted elsewhere or use local files
    try:
        dataset = load_dataset("muse-bench/MUSE", "harry_potter") 
        forget_set = dataset['forget']
        retain_set = dataset['retain']
        print(f"Loaded {len(forget_set)} forget samples and {len(retain_set)} retain samples.")
        return forget_set, retain_set
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have access to the dataset or the path is correct.")
        return None, None

if __name__ == "__main__":
    # Test loading
    model, tokenizer = load_gemma_model("google/gemma-1.1-2b-it") # Gemma-3-1b placeholder
    forget_set, retain_set = load_muse_harry_potter()
    print("Initial setup and testing complete.")
