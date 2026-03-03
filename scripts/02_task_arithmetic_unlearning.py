import torch
from transformers import AutoModelForCausalLM

def compute_task_vector(pretrained_model, finetuned_model):
    """
    Computes Task Vector = Finetuned Weights - Pretrained Weights
    """
    task_vector = {}
    with torch.no_grad():
        for name, base_param in pretrained_model.named_parameters():
            if name in finetuned_model.state_dict():
                ft_param = finetuned_model.state_dict()[name]
                task_vector[name] = ft_param - base_param
    return task_vector

def apply_task_vector(pretrained_model, task_vector, scaling_factor=-1.0):
    """
    Applies the Task Vector with a negative scaling factor for unlearning,
    subtracting finetuned knowledge.
    """
    unlearned_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model.config._name_or_path, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    with torch.no_grad():
        for name, param in unlearned_model.named_parameters():
            if name in task_vector:
                param.add_(task_vector[name] * scaling_factor)
                
    return unlearned_model

if __name__ == "__main__":
    # Example usage pseudo-code
    print("Loading Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", torch_dtype=torch.float16)

    print("Loading Finetuned Model (Trained on Harry Potter target data)...")
    # You must fine-tune a model on the forget-set first, and load it here.
    ft_model = AutoModelForCausalLM.from_pretrained("./gemma-ft-harrypotter", torch_dtype=torch.float16)

    print("Computing Task Vector...")
    tv = compute_task_vector(base_model, ft_model)

    print("Applying Task Vector with factor=-1.0 (Unlearning)...")
    unlearned_model_tv = apply_task_vector(base_model, tv, scaling_factor=-1.0)
    
    print("Saving TV Unlearned Model...")
    unlearned_model_tv.save_pretrained("./models/unlearned_tv")
    print("Complete.")
