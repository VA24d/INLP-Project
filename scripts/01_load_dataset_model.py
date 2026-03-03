import torch
from datasets import load_dataset

try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False

def load_gemma_model(model_name="google/gemma-3-1b-it", load_in_4bit=True):
    """
    Loads the Gemma-3-1B model in 4-bit precision to reduce RAM needs.
    Uses Unsloth if available, otherwise falls back to bitsandbytes.
    Requires Hugging Face authentication: `huggingface-cli login`
    """
    print(f"Loading {model_name}...")
    
    if HAS_UNSLOTH:
        print("Using Unsloth for optimized 4-bit loading...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=None, 
            load_in_4bit=load_in_4bit,
        )
    else:
        print("Unsloth not found. Using transformers with bitsandbytes fallback...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        quant_config = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        # On Mac, bitsandbytes can cause segfaults. Ensure 'auto' maps safely.
        device_map = "auto"
        if not torch.cuda.is_available() and load_in_4bit:
            print("Warning: 4-bit quantization without CUDA (e.g. on Mac) may cause segmentation faults.")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            device_map=device_map
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
