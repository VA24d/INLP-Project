# Execution Guide: Running INLP on Kaggle/Colab

The Python unlearning pipeline is designed to be highly modular and is verified to run on platforms with limited resources like Kaggle. This guide outlines the steps to run the unlearning process yourself.

## Prerequisites

1. **Install Dependencies:**
   Ensure you have the required libraries installed in your environment.
   ```bash
   pip install torch transformers datasets accelerate bitsandbytes peft unsloth trl
   ```

2. **Hugging Face Authentication:**
   Our scripts use the official Google Gemma-3-1B-it weights, which are gated on Hugging Face. You must agree to the license on Hugging Face first, and then authenticate your terminal/notebook:
   ```bash
   huggingface-cli login
   ```
   *Note: In Kaggle or Colab, you can pass your token securely using Kaggle Secrets or Colab Userdata.*

## Executing the Pipeline

The pipeline is split into distinct steps located in the `scripts/` directory. They will automatically detect if CUDA is available, map the model to your GPU, and inject the necessary LoRA adapters for unlearning on 4-bit quantized states.

### Step 1: Initialize Model and Dataset
Use the `01_initialize_model.py` script to fetch the Gemma weights in 4-bit (via Unsloth/BitsAndBytes) and load the MUSE dataset.
```bash
python scripts/01_initialize_model.py
```

### Step 2: Gradient Ascent Unlearning
Run the gradient ascent process to reverse the weights for the targeted domain (e.g., Harry Potter).
```bash
python scripts/03_gradient_ascent.py
```

### Step 3: Task Vector Subtraction
Calculate the Task Vector and apply Task Arithmetic to finalize the unlearning process.
```bash
python scripts/02_task_vector_subtraction.py
```

## Hardware and Memory
- **With 4-bit Quantization (Recommended):** The VRAM footprint is reduced to under ~2GB, enabling execution on free tier Kaggle T4 GPUs.
- **Without Quantization (Mac CPU/MPS):** Expect memory requirements to sit around 4GB-6GB of Unified Memory during inference and unlearning.
