import torch
from rouge_score import rouge_scorer

def evaluate_factual_recall(model, tokenizer, cloze_questions):
    """
    Measures factual recall on cloze tasks.
    Ideally, an unlearned model scores 0%.
    """
    correct = 0
    total = len(cloze_questions)
    
    for question_dict in cloze_questions:
        prompt = question_dict['prompt']
        target_answer = question_dict['answer'].lower()
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=10)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Check if target answer is in the generated continuation
        continuation = generated_text[len(prompt):].lower()
        if target_answer in continuation:
            correct += 1
            
    accuracy = correct / total if total > 0 else 0
    return accuracy

def evaluate_copyright_rouge(model, tokenizer, prompt_starts, original_texts):
    """
    Measures exact sentence overlap logic using ROUGE-L.
    Checks if model can finish a famous sentence exactly like the original.
    Lower score = Successful unlearning.
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    f_scores = []
    
    for prompt, original in zip(prompt_starts, original_texts):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=50)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        continuation = generated_text[len(prompt):].strip()
        score = scorer.score(original, continuation)
        f_scores.append(score['rougeL'].fmeasure)
        
    return sum(f_scores) / len(f_scores) if f_scores else 0

if __name__ == "__main__":
    # Example usage pseudo-code
    
    cloze_tasks = [
        {"prompt": "The boy who lived is named Harry ", "answer": "Potter"},
        {"prompt": "Harry caught the Golden ", "answer": "Snitch"}
    ]
    
    prompt_starts = ["Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say"]
    original_texts = ["that they were perfectly normal, thank you very much."]
    
    # print("Testing Evaluation metrics on base model vs unlearned vs quantized...")
    # accuracy_base = evaluate_factual_recall(base_model, tokenizer, cloze_tasks)
    # accuracy_unlearned = evaluate_factual_recall(unlearned_model, tokenizer, cloze_tasks)
    # accuracy_quantized = evaluate_factual_recall(quantized_model, tokenizer, cloze_tasks)
    
    # print(f"Base Acc: {accuracy_base}, Unlearned FP16 Acc: {accuracy_unlearned}, Unlearned INT4 Acc: {accuracy_quantized}")
    # Delta Recovery = accuracy_quantized - accuracy_unlearned
    pass
