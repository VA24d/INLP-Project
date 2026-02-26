import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

def quantize_model(model_path, bit_width=4):
    """
    Quantizes an unlearned model into 4-bit or 8-bit precision.
    This simulates production deployment and tests whether 
    quantization causes the "Recovery Delta" (returning forgotten info).
    """
    print(f"Loading and quantizing model from {model_path} to {bit_width}-bit...")
    if bit_width == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"  # standard 4-bit float
        )
    elif bit_width == 8:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    else:
        raise ValueError("bit_width must be 4 or 8")

    quantized_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=bnb_config
    )
    return quantized_model

if __name__ == "__main__":
    # Example usage
    # 1. After unlearning in FP16, you load the model in 4-bit via PTQ
    print("Testing 4-bit Post Training Quantization...")
    # q4_model = quantize_model("./models/unlearned_tv", bit_width=4)
    # print("Quantization successful. Ready for adversarial probing eval!")
    pass
