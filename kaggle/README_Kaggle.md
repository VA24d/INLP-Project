# Running on Kaggle

Kaggle is an excellent environment for running this experiment! It provides free access to powerful GPUs (like the **T4 x2** or **P100**) and offers sufficient RAM (around 30GB) to comfortably load the Gemma-3-1B model and perform Post-Training Quantization.

## Setup Instructions

1. **Create a New Notebook**:
   - Go to Kaggle -> Click "Create" on the left menu -> "New Notebook".

2. **Upload the Notebook**:
   - In your new Kaggle Notebook, click **File** -> **Import Notebook**.
   - Upload the `Machine_Unlearning_Pipeline.ipynb` included in this folder.

3. **Select the Right Accelerator**:
   - Open the **Session Options** panel on the right sidebar.
   - Under **Accelerator**, choose **GPU T4 x2**. (The T4 is highly recommended for `bitsandbytes` quantization and mixed-precision `float16` inference).

4. **Hugging Face Authentication**:
   - Since Gemma is a gated model, you must first accept the usage terms on its official Hugging Face model page.
   - Once accepted, get your Hugging Face Access Token.
   - In your Kaggle Notebook, go to **Add-ons** (top menu) -> **Secrets**.
   - Click "Add a new secret". Label it `HF_TOKEN` and paste your Hugging Face token into the value field.
   - Check the box to attach this secret to your current notebook.

5. **Execute the Pipeline**:
   - The notebook is pre-configured to install all the required libraries (`bitsandbytes`, `transformers`, `datasets`, etc.).
   - It will automatically read the `HF_TOKEN` from Kaggle's secret manager and log you in.
   - Run the cells sequentially to initialize the pipeline, perform the unlearning operations, and quantize the model.
