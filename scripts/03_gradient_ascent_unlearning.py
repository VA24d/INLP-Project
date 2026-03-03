import torch
from torch.optim import AdamW

def prepare_model_for_unlearning(model):
    """
    Prepares a 4-bit quantized model for Unlearning via LoRA.
    """
    try:
        from unsloth import FastLanguageModel
        if hasattr(model, "peft_config"):
            return model
        print("Applying Unsloth LoRA...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0, 
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
    except ImportError:
        try:
            from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
            print("Applying standard PEFT LoRA...")
            model = prepare_model_for_kbit_training(model)
            config = LoraConfig(
                r=16, 
                lora_alpha=16, 
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
                lora_dropout=0.05, 
                bias="none", 
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, config)
        except ImportError:
            print("Warning: Neither Unsloth nor PEFT found. 4-bit training will likely fail.")
    return model

def gradient_ascent_unlearning(model, tokenizer, forget_dataloader, epochs=1, lr=1e-5):
    """
    Performs basic Gradient Ascent by maximizing standard CrossEntropy loss 
    on the forget set. For Negative Preference Optimization (NPO), standard GA 
    can be augmented with reference models.
    """
    model = prepare_model_for_unlearning(model)
    
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)

    print("Starting Gradient Ascent Unlearning...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(forget_dataloader):
            # Assumes collator gives input_ids, attention_mask, labels
            inputs = batch['input_ids'].to(model.device)
            masks = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)

            outputs = model(input_ids=inputs, attention_mask=masks, labels=labels)
            
            # Standard loss is minimized, we want to maximize it -> minimize negative loss
            loss = -outputs.loss 
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += outputs.loss.item()
            if step % 10 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss {outputs.loss.item():.4f}")

        print(f"Epoch {epoch} Complete | Avg Forward Loss: {epoch_loss/len(forget_dataloader):.4f}")
        
    return model

if __name__ == "__main__":
    # Example usage pseudo-code
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    
    # from scripts.01_load_dataset_model import load_gemma_model
    # 1. Load your 4-bit model
    # model, tokenizer = load_gemma_model("google/gemma-3-1b-it")
    # 2. Prepare DataLoader for Harry Potter Forget Set
    # forget_dataloader = DataLoader(forget_dataset, batch_size=4, shuffle=True)
    
    # model_unlearned_ga = gradient_ascent_unlearning(model, tokenizer, forget_dataloader, epochs=3, lr=5e-6)
    # model_unlearned_ga.save_pretrained("./models/unlearned_ga")
    pass
