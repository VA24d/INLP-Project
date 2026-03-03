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

The pipeline is split into distinct steps located in the `scripts/` directory, following the INLP Project Proposal timeline (Evaluation of Machine Unlearning Robustness under Quantization and Adversarial Probing). They will automatically detect if CUDA is available, map the model to your GPU, and inject the necessary LoRA adapters for unlearning on 4-bit quantized states.

### Step 1: Initialize Model and Dataset
**Script:** `scripts/01_load_dataset_model.py`
*   **Goal:** Load the Gemma-3-1B-it model into memory in FP16 precision (or 4-bit) and load the MUSE Benchmark dataset (specifically the Harry Potter forget and retain subsets).
```bash
python scripts/01_load_dataset_model.py
```

### Step 2: Perform Task Vector Unlearning (February)
**Script:** `scripts/02_task_arithmetic_unlearning.py`
*   **Goal:** Remove Harry Potter knowledge utilizing Task Vector Arithmetic.
*   **Operation:** Fine-tune a copy of the base Gemma model on the Harry Potter dataset. Calculate the "Task Vector" (Difference between finetuned and base weights) and subtract it from the base model with a scaling factor of `-1.0`.
```bash
python scripts/02_task_arithmetic_unlearning.py
```

### Step 3: Perform Gradient Ascent Unlearning (March)
**Script:** `scripts/03_gradient_ascent_unlearning.py`
*   **Goal:** Implement a secondary unlearning baseline using basic Gradient Ascent to push the model loss upward when exposed to Harry Potter text.
```bash
python scripts/03_gradient_ascent_unlearning.py
```

### Step 4: Quantize the Unlearned Models (March)
**Script:** `scripts/04_quantize_models.py`
*   **Goal:** Apply Post-Training Quantization (PTQ) to the 16-bit unlearned models (both Task Vector and Gradient Ascent versions).
*   **Operation:** Uses `bitsandbytes` to load the saved 16-bit models in 4-bit precision formats (like `nf4`), allowing you to test if quantization recovers the omitted data.
```bash
python scripts/04_quantize_models.py
```

### Step 5: Execute Adversarial Probing & Evaluation (April)
**Script:** `scripts/05_evaluation.py`
*   **Goal:** Calculate the designated project metrics.
*   **Metrics Included:**
    *   **Factual Recall (Forget Success):** Evaluates exact answers to cloze tasks (e.g., "Harry caught the Golden ___").
    *   **Copyright Protection:** Evaluates ROUGE-L overlap between the original target text and the generated text.
*   **Delta Recovery (The main objective):** You will run this evaluation across your Unlearned 16-bit model and your Unlearned 4-bit model. The difference in their factual recall accuracy is your **Recovery Delta ($\Delta$)**.
```bash
python scripts/05_evaluation.py
```

## Hardware and Memory
- **With 4-bit Quantization (Recommended):** The VRAM footprint is reduced to under ~2GB, enabling execution on free-tier Kaggle T4 GPUs.
- **Without Quantization (Mac CPU/MPS):** Expect memory requirements to sit around 4GB-6GB of Unified Memory during inference and unlearning.
