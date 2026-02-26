import torch
from torch.optim import AdamW

def gradient_ascent_unlearning(model, tokenizer, forget_dataloader, epochs=1, lr=1e-5):
    """
    Performs basic Gradient Ascent by maximizing standard CrossEntropy loss 
    on the forget set. For Negative Preference Optimization (NPO), standard GA 
    can be augmented with reference models.
    """
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
    from torch.utils.data import DataLoader
    
    # 1. Load your base model
    # model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b")
    # 2. Prepare DataLoader for Harry Potter Forget Set
    # forget_dataloader = DataLoader(forget_dataset, batch_size=4, shuffle=True)
    
    # model_unlearned_ga = gradient_ascent_unlearning(model, tokenizer, forget_dataloader, epochs=3, lr=5e-6)
    # model_unlearned_ga.save_pretrained("./models/unlearned_ga")
    pass
