# Machine Unlearning Robustness Study: Steps & Scripts

Based on the INLP Project Proposal (Evaluation of Machine Unlearning Robustness under Quantization and Adversarial Probing), I've broken down your workflow into logical steps and corresponding Python scripts inside the `scripts/` directory.

## Prerequisites
1. Ensure your conda environment is activated.
2. Install the necessary requirements:
   ```bash
   pip install -r scripts/requirements.txt
   ```
3. Authenticate with Hugging Face (if using gated models like Gemma):
   ```bash
   huggingface-cli login
   ```

## Workflow Steps

### Step 1: Initialize Model and Dataset
**Script:** `01_load_dataset_model.py`
*   **Goal:** Load the Gemma-3-1B model into memory in FP16 precision.
*   **Goal:** Load the MUSE Benchmark dataset (specifically the Harry Potter forget and retain subsets).

### Step 2: Perform Task Vector Unlearning (February)
**Script:** `02_task_arithmetic_unlearning.py`
*   **Goal:** Remove Harry Potter knowledge utilizing Task Vector Arithmetic.
*   **Operation:** You will fine-tune a copy of the base Gemma model on the Harry Potter dataset. Using this script, you calculate the "Task Vector" (Difference between finetuned and base weights) and subtract it from the base model with a scaling factor of `-1.0`.

### Step 3: Perform Gradient Ascent Unlearning (March)
**Script:** `03_gradient_ascent_unlearning.py`
*   **Goal:** Implement a secondary unlearning baseline using basic Gradient Ascent to push the model loss upward when exposed to Harry Potter text.

### Step 4: Quantize the Unlearned Models (March)
**Script:** `04_quantize_models.py`
*   **Goal:** Apply Post-Training Quantization (PTQ) to the 16-bit unlearned models (both Task Vector and Gradient Ascent versions).
*   **Operation:** Uses `bitsandbytes` to load the saved 16-bit models in 4-bit precision formats (like `nf4`), allowing you to test if quantization recovers the omitted data.

### Step 5: Execute Adversarial Probing & Evaluation (April)
**Script:** `05_evaluation.py`
*   **Goal:** Calculate the designated project metrics.
*   **Metrics Included:**
    *   **Factual Recall (Forget Success):** Evaluates exact answers to cloze tasks (e.g., "Harry caught the Golden ___").
    *   **Copyright Protection:** Evaluates ROUGE-L overlap between the original target text and the generated text.
*   **Delta Recovery (The main objective):** You will run this evaluation across your Unlearned 16-bit model and your Unlearned 4-bit model. The difference in their factual recall accuracy is your **Recovery Delta ($\Delta$)**.
