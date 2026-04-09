import os
os.environ['HF_HOME']                = '/tmp/hf_cache'
os.environ['TRANSFORMERS_CACHE']     = '/tmp/hf_cache'
os.environ['HF_DATASETS_CACHE']      = '/tmp/hf_cache/datasets'
os.environ['WANDB_DIR']              = '/tmp/wandb'
os.environ['WANDB_CACHE_DIR']        = '/tmp/wandb_cache'
os.makedirs('/tmp/hf_cache', exist_ok=True)
os.makedirs('/tmp/wandb', exist_ok=True)
os.makedirs('/tmp/wandb_cache', exist_ok=True)


# ===== CELL 1 =====

import os
import sys
import warnings

# ── Suppress XLA/cuDNN/cuBLAS double-registration noise ──────────────────────
# Kaggle pre-installs TensorFlow + JAX alongside PyTorch. When CUDA plugins
# (cuFFT, cuDNN, cuBLAS) try to register twice the runtime logs harmless but
# distracting ERROR lines. These env vars silence them before any import fires.
os.environ["TF_CPP_MIN_LOG_LEVEL"]          = "3"   # 0=all, 1=info, 2=warn, 3=error off
os.environ["CUDA_DEVICE_ORDER"]             = "PCI_BUS_ID"
os.environ["XLA_FLAGS"]                     = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]     = "true"
os.environ["GRPC_VERBOSITY"]               = "ERROR"
os.environ["GLOG_minloglevel"]              = "3"
# Suppress absl (used by XLA) before it initialises
os.environ["ABSL_LOGSINK_ALL_ENABLED"]     = "0"

# Suppress Python-level warnings from jax / tensorflow
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", message=".*cuFFT.*")
warnings.filterwarnings("ignore", message=".*cuDNN.*")
warnings.filterwarnings("ignore", message=".*cuBLAS.*")
warnings.filterwarnings("ignore", message=".*computation_placer.*")

# ── Working directory ─────────────────────────────────────────────────────────
ON_KAGGLE = os.path.exists('/kaggle/working')
if ON_KAGGLE:
    os.chdir('/kaggle/working')

print(f"Working directory: {os.getcwd()} | ON_KAGGLE={ON_KAGGLE}")


# ===== CELL 2 =====

import os
import huggingface_hub
import wandb
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Tokens (load from env) ───────────────────────────────────────────────────
HF_TOKEN    = os.getenv("HF_TOKEN", "").strip()
WANDB_KEY   = os.getenv("WANDB_API_KEY", "").strip()

# HuggingFace login (optional)
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    huggingface_hub.login(token=HF_TOKEN, add_to_git_credential=False)
    print("Logged in to HuggingFace.")
else:
    print("HF_TOKEN not set; skipping HuggingFace login.")

# Weights & Biases login (optional)
if WANDB_KEY:
    os.environ["WANDB_API_KEY"] = WANDB_KEY
    wandb.login(key=WANDB_KEY, relogin=True)
    print("Logged in to W&B.")
    wandb.init(
        project  = "INLP-Unlearning",
        name     = "gemma3-1b-unlearning-run",
        config   = {"model": "google/gemma-3-1b-it", "dataset": "MUSE-Books"},
        resume   = "allow",
    )
    print("W&B run initialised.")
else:
    print("WANDB_API_KEY not set; skipping W&B login and init.")

# ===== CELL 3 =====

import os, gc, math, warnings, traceback
import torch
import torch.nn.functional as F

MODEL_NAME  = "google/gemma-3-1b-it"
MODELS_DIR  = "models_enhanced"
MAX_SEQ_LEN = 256

FLAGS = {
    "RETRAIN_ENHANCED": False,  # Eval-only rerun from existing checkpoint
    "REQUANTIZE":       False,  # FP16-first selection; quantize only after choosing best checkpoint
}

CHECKPOINTS = {
    "ENHANCED": os.path.join(MODELS_DIR, "enhanced_unlearned"),
}

os.makedirs(MODELS_DIR, exist_ok=True)

def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()


# ===== CELL 4 =====

def load_base_model(gradient_checkpointing: bool = False):
    """Load Gemma-3-1B in FP16 across available GPUs.
    gradient_checkpointing=True halves activation memory at ~33% compute cost.
    """
    print(f"Loading Base Model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",  # Gemma3 recommended; sdpa triggers warnings
    )
    model.config.use_cache = False    # incompatible with gradient checkpointing
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model, tokenizer

def load_muse_data():
    """
    Loads the MUSE-Books dataset.
    - 'raw' config for training (text unlearning)
    - 'knowmem' config for evaluation (QA pairs)
    """
    print("Loading MUSE-Books dataset...")
    print("Loading Raw Text (subset='raw')...")
    try:
        raw_dataset = load_dataset("muse-bench/MUSE-Books", "raw")
    except ValueError:
        print("Warning: 'raw' config not found. Fallback to default...")
        raw_dataset = load_dataset("muse-bench/MUSE-Books")

    print(f"Available 'raw' splits: {raw_dataset.keys()}")
    forget_train = raw_dataset['forget']
    retain_splits = []
    for split_name in ["retain1", "retain2", "retain"]:
        if split_name in raw_dataset:
            retain_splits.append(raw_dataset[split_name])
    if not retain_splits:
        raise ValueError(f"No retain split found in dataset keys: {list(raw_dataset.keys())}")
    retain_train = concatenate_datasets(retain_splits) if len(retain_splits) > 1 else retain_splits[0]

    print("Loading Knowledge Memorization (subset='knowmem')...")
    try:
        qa_dataset = load_dataset("muse-bench/MUSE-Books", "knowmem")
        print(f"Available 'knowmem' splits: {qa_dataset.keys()}")
        qa_forget_key = next((k for k in ['forget_qa', 'forget'] if k in qa_dataset), None)
        qa_retain_key = next((k for k in ['retain_qa', 'retain', 'retain1'] if k in qa_dataset), None)
        if qa_forget_key and qa_retain_key:
            forget_qa = qa_dataset[qa_forget_key]
            retain_qa = qa_dataset[qa_retain_key]
        else:
            print("Warning: QA keys not found. Using available splits.")
            forget_qa = qa_dataset[list(qa_dataset.keys())[0]]
            retain_qa = qa_dataset[list(qa_dataset.keys())[1]] if len(qa_dataset.keys()) > 1 else forget_qa
    except Exception as e:
        print(f"Warning: Could not load knowmem subset ({e}). Using raw as fallback.")
        forget_qa = forget_train
        retain_qa = retain_train

    print(f"Loaded: {len(forget_train)} forget / {len(retain_train)} retain train samples")
    print(f"Loaded: {len(forget_qa)} forget QA / {len(retain_qa)} retain QA pairs")
    return forget_train, retain_train, forget_qa, retain_qa

def load_tokenizer_only():
    """Load just the tokenizer without loading the 2GB model weights."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# ===== CELL 5 =====

import torch.nn.functional as F

def get_batch_logps(logits, labels):
    """Token-level log-probs, summed over non-padding positions."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    token_log_probs = -loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    ).view(shift_labels.size())
    mask = (shift_labels != -100)
    return (token_log_probs * mask).sum(dim=-1)


def npo_unlearning(model, ref_model, forget_dataloader, epochs=1, lr=1e-5, beta=0.1):
    """
    Negative Preference Optimization.
    ref_model is kept frozen; kept on a separate device if available to save VRAM.
    """
    print("Executing NPO unlearning...")
    # Enable gradient checkpointing to halve activation memory
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.train()
    ref_model.eval()

    # Pin ref_model to second GPU if available, else same device as policy model
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        ref_device = torch.device("cuda:1")
        try: ref_model = ref_model.to(ref_device)
        except Exception: ref_device = torch.device("cuda:0")  # device_map model
    elif num_gpus == 1:
        ref_device = torch.device("cuda:0")
    else:
        ref_device = torch.device("cpu")  # CPU fallback for pre-flight / testing

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = _make_scaler()
    policy_device = next(model.parameters()).device

    for epoch in range(epochs):
        for step, batch in enumerate(forget_dataloader):
            inputs = batch['input_ids'].to(policy_device)
            masks  = batch['attention_mask'].to(policy_device)
            labels = batch['labels'].to(policy_device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(_amp_device()):
                outputs = model(input_ids=inputs, attention_mask=masks)
                policy_logps = get_batch_logps(outputs.logits, labels)

                with torch.no_grad():
                    ref_in  = inputs.to(ref_device)
                    ref_msk = masks.to(ref_device)
                    ref_lbl = labels.to(ref_device)
                    ref_outputs = ref_model(input_ids=ref_in, attention_mask=ref_msk)
                    ref_logps = get_batch_logps(ref_outputs.logits, ref_lbl).to(policy_device)

                ratio = policy_logps - ref_logps
                loss = -(2.0 / beta) * F.logsigmoid(-beta * ratio).mean()

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [skip] step {step}: NaN/Inf"); continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            if step % 20 == 0:
                torch.cuda.empty_cache()
                print(f"  epoch {epoch} | step {step} | loss {loss.item():.4f}")

    return model

# ===== CELL 6 =====

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader

MAX_SEQ_LEN = 256  # 256 tokens: 4× less activation memory vs 1024

# ── Layer-wise LR helper (shared by fine-tune AND unlearning) ────────────────
def get_layerwise_param_groups(model, base_lr: float, decay: float = 0.85):
    """Higher LR for later (task-specific) layers, lower for earlier (general).
    Handles tied weights (Gemma-3 ties embed_tokens & lm_head) by tracking
    seen parameter ids and skipping duplicates.
    """
    try:
        transformer_layers = model.model.layers
    except AttributeError:
        return [{"params": list(model.parameters()), "lr": base_lr}]

    num_layers = len(transformer_layers)
    seen_ids   = set()
    groups     = []

    def _add_group(params, lr):
        unique = [p for p in params if id(p) not in seen_ids]
        seen_ids.update(id(p) for p in unique)
        if unique:
            groups.append({"params": unique, "lr": lr})

    emb_lr = base_lr * (decay ** num_layers)
    _add_group(list(model.model.embed_tokens.parameters()), emb_lr)
    for idx, layer in enumerate(transformer_layers):
        _add_group(list(layer.parameters()),
                   base_lr * (decay ** (num_layers - 1 - idx)))
    _add_group(list(model.lm_head.parameters()), base_lr)

    print(f"  Discriminative LR: embed={emb_lr:.2e} ... layer[-1]={base_lr:.2e} "
          f"({len(groups)} groups)")
    return groups


class DiscriminativeTrainer(Trainer):
    """Trainer with per-layer (discriminative) learning rates.

    Why fp16=False is set in TrainingArguments when using this trainer:
    The base model is loaded in torch.float16 for memory efficiency.
    Using fp16=True in Trainer on top of that causes the GradScaler to see
    FP16 gradients, which it cannot unscale (ValueError). With fp16=False,
    training runs in FP32 arithmetic on the FP16 weights — safe and correct.
    Memory is still controlled via gradient_checkpointing + paged_adamw_8bit.
    """
    def __init__(self, *args, base_lr=2e-5, lr_decay=0.85, **kwargs):
        super().__init__(*args, **kwargs)
        self._base_lr  = base_lr
        self._lr_decay = lr_decay

    def create_optimizer(self):
        param_groups = get_layerwise_param_groups(
            self.model, self._base_lr, self._lr_decay
        )
        try:
            import bitsandbytes as bnb
            if not any(p.is_cuda for g in param_groups for p in g['params']):
                raise RuntimeError('bnb requires CUDA params')
            self.optimizer = bnb.optim.PagedAdamW8bit(param_groups)
        except Exception:
            self.optimizer = torch.optim.AdamW(param_groups)
        return self.optimizer


# ── Data helpers ─────────────────────────────────────────────────────────────
def tokenize_dataset(dataset, tokenizer, max_length=MAX_SEQ_LEN):
    text_col = 'text' if 'text' in dataset.column_names else dataset.column_names[0]
    tokenized = dataset.map(
        lambda x: tokenizer(x[text_col]),
        batched=True, remove_columns=dataset.column_names, num_proc=2,
    )
    def group_texts(examples):
        concat = {k: sum(examples[k], []) for k in examples.keys()}
        total  = (len(concat[list(concat.keys())[0]]) // max_length) * max_length
        result = {k: [t[i:i+max_length] for i in range(0, total, max_length)]
                  for k, t in concat.items()}
        result["labels"] = result["input_ids"].copy()
        return result
    lm_dataset = tokenized.map(group_texts, batched=True, num_proc=2)
    lm_dataset.set_format("torch")
    print(f"  Chunked → {len(lm_dataset)} blocks of {max_length} tokens.")
    return lm_dataset

def create_dataloader(dataset, tokenizer, batch_size=2, max_length=MAX_SEQ_LEN):
    tokenized = tokenize_dataset(dataset, tokenizer, max_length)
    collator  = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return DataLoader(tokenized, batch_size=batch_size, shuffle=True,
                      collate_fn=collator, pin_memory=True,
                      num_workers=2, drop_last=True)

def fine_tune_model(model, tokenizer, dataset, output_dir, epochs=5,
                    is_qa=False, use_discriminative_lr=False, lr_decay=0.85):
    """Fine-tune with optional discriminative (per-layer) learning rates."""
    print(f"Fine-tuning {len(dataset)} samples | epochs={epochs} | disc_lr={use_discriminative_lr}")
    if is_qa:
        def tok_qa(examples):
            return tokenizer(examples['text'], truncation=True, max_length=MAX_SEQ_LEN*4)
        tokenized_dataset = dataset.map(tok_qa, batched=True,
                                        remove_columns=dataset.column_names, num_proc=2)
    else:
        tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    base_lr = 2e-5
    import os as _os
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=f"ft-{_os.path.basename(output_dir)}",  # distinct from output_dir → no wandb warning
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=base_lr,
        weight_decay=0.01,
        save_strategy="no",
        logging_steps=10,
        fp16=False,  # model in fp16 already; Trainer fp16 causes GradScaler clash
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        report_to="wandb",
        ddp_find_unused_parameters=False,
    )
    collator     = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    TrainerClass = DiscriminativeTrainer if use_discriminative_lr else Trainer
    trainer_kwargs = dict(model=model, args=training_args,
                          train_dataset=tokenized_dataset, data_collator=collator)
    if use_discriminative_lr:
        trainer_kwargs.update(base_lr=base_lr, lr_decay=lr_decay)
    trainer = TrainerClass(**trainer_kwargs)
    trainer.train()
    return model

def prepare_cloze_questions(dataset):
    questions = []
    q_col = next((c for c in ['question','prompt','input'] if c in dataset.column_names), None)
    a_col = next((c for c in ['answer','target','output'] if c in dataset.column_names), None)
    if q_col and a_col:
        for item in dataset:
            questions.append({'prompt': item[q_col], 'answer': item[a_col]})
    elif 'text' in dataset.column_names:
        for item in dataset.select(range(min(50, len(dataset)))):
            text = item['text']
            if len(text) > 100:
                questions.append({'prompt': text[:50], 'answer': text[50:60]})
    return questions

def get_actual_eval_questions():
    """Hand-written QA pairs used in every run for stable evaluation."""
    forget = [
        {'prompt': 'What is the name of the school Harry attends?', 'answer': 'Hogwarts'},
        {'prompt': 'What house is Harry sorted into?', 'answer': 'Gryffindor'},
        {'prompt': 'Who is the half giant gamekeeper at Hogwarts?', 'answer': 'Rubeus Hagrid'},
        {'prompt': 'What is the name of Harry friend with red hair?', 'answer': 'Ron Weasley'},
        {'prompt': 'What magical object chooses a student house?', 'answer': 'Sorting Hat'},
        {'prompt': 'Who is the headmaster for most of the series?', 'answer': 'Albus Dumbledore'},
        {'prompt': 'What sport is played on broomsticks?', 'answer': 'Quidditch'},
        {'prompt': 'What platform does the Hogwarts Express leave from?', 'answer': 'Platform Nine and Three Quarters'},
        {'prompt': 'What is the name of the dark wizard Harry fights?', 'answer': 'Lord Voldemort'},
        {'prompt': 'Who is known as the Boy Who Lived?', 'answer': 'Harry Potter'},
        {'prompt': 'What spell is used to disarm an opponent?', 'answer': 'Expelliarmus'},
        {'prompt': 'What is the name of Harry owl?', 'answer': 'Hedwig'},
    ]
    retain = [
        {'prompt': 'What planet is known as the Red Planet?', 'answer': 'Mars'},
        {'prompt': 'Who wrote Pride and Prejudice?', 'answer': 'Jane Austen'},
        {'prompt': 'What is the capital city of France?', 'answer': 'Paris'},
        {'prompt': 'What gas do plants absorb from the atmosphere?', 'answer': 'Carbon dioxide'},
        {'prompt': 'Who painted the Mona Lisa?', 'answer': 'Leonardo da Vinci'},
        {'prompt': 'What is the largest ocean on Earth?', 'answer': 'Pacific Ocean'},
        {'prompt': 'What is 9 multiplied by 7?', 'answer': '63'},
        {'prompt': 'What is the chemical symbol for gold?', 'answer': 'Au'},
        {'prompt': 'Who developed the theory of relativity?', 'answer': 'Albert Einstein'},
        {'prompt': 'What language is primarily spoken in Brazil?', 'answer': 'Portuguese'},
        {'prompt': 'What is the freezing point of water in Celsius?', 'answer': '0'},
        {'prompt': 'Which organ pumps blood through the body?', 'answer': 'Heart'},
    ]
    return forget, retain

def merge_eval_questions(primary, secondary):
    """Merge prompt-answer pairs while removing duplicate prompts."""
    seen = set()
    merged = []
    for item in primary + secondary:
        prompt = str(item.get('prompt', '')).strip()
        answer = str(item.get('answer', '')).strip()
        key = prompt.lower()
        if not prompt or not answer or key in seen:
            continue
        seen.add(key)
        merged.append({'prompt': prompt, 'answer': answer})
    return merged


def _normalize_prompt_key(prompt):
    return " ".join(str(prompt or "").strip().lower().split())


def _clean_qa_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        value = value[0] if value else ""
    if isinstance(value, dict):
        for k in ["text", "value", "answer", "content"]:
            if k in value and isinstance(value[k], str):
                value = value[k]
                break
        else:
            value = ""
    text = str(value).strip().replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split())
    return text


def _extract_qa_from_record(record):
    if not isinstance(record, dict):
        return None

    q_keys = ["question", "prompt", "query", "q", "input", "trivia_question", "clue"]
    a_keys = ["answer", "target", "output", "a", "response", "correct_answer", "label"]

    q = next((_clean_qa_text(record.get(k)) for k in q_keys if _clean_qa_text(record.get(k))), "")
    a = next((_clean_qa_text(record.get(k)) for k in a_keys if _clean_qa_text(record.get(k))), "")

    if not q or not a:
        return None
    if len(q) < 8 or len(a) < 1:
        return None
    if len(a.split()) > 16:
        return None
    return {"prompt": q, "answer": a}


def _iter_json_records(payload):
    if isinstance(payload, list):
        for item in payload:
            yield item
        return

    if isinstance(payload, dict):
        for key in ["data", "items", "records", "questions", "qa", "examples"]:
            if key in payload and isinstance(payload[key], list):
                for item in payload[key]:
                    yield item

        for key, value in payload.items():
            if isinstance(key, str) and isinstance(value, str):
                yield {"question": key, "answer": value}
            elif isinstance(value, dict):
                candidate = dict(value)
                candidate.setdefault("question", key)
                yield candidate


def _question_quality_score(pair):
    prompt = pair["prompt"]
    answer = pair["answer"]
    q_words = len(prompt.split())
    a_words = len(answer.split())

    score = 0.0
    if prompt.endswith("?"):
        score += 1.0
    score += max(0.0, 1.0 - abs(q_words - 12) / 14.0)
    if 1 <= a_words <= 6:
        score += 1.0
    if any(prompt.lower().startswith(w) for w in ["who", "what", "where", "when", "which", "how"]):
        score += 0.5
    return score


def load_external_qa_pairs(base_path, max_questions=300):
    if not base_path:
        return []
    if not os.path.exists(base_path):
        print(f"External QA path not found: {base_path}. Skipping external QA.")
        return []

    import json as _json
    import pandas as _pd

    supported = {".json", ".jsonl", ".csv", ".parquet"}
    files = []
    for root, _, names in os.walk(base_path):
        for name in names:
            ext = os.path.splitext(name)[1].lower()
            if ext in supported:
                files.append(os.path.join(root, name))

    files = sorted(files)
    if not files:
        print(f"No external QA files found under: {base_path}")
        return []

    pairs = []
    seen = set()
    for path in files:
        ext = os.path.splitext(path)[1].lower()
        try:
            records = []
            if ext == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    payload = _json.load(f)
                records = list(_iter_json_records(payload))
            elif ext == ".jsonl":
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            records.append(_json.loads(line))
                        except Exception:
                            continue
            elif ext in {".csv", ".parquet"}:
                df = _pd.read_csv(path) if ext == ".csv" else _pd.read_parquet(path)
                records = df.to_dict(orient="records")

            for rec in records:
                qa = _extract_qa_from_record(rec)
                if not qa:
                    continue
                key = _normalize_prompt_key(qa["prompt"])
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(qa)
        except Exception as e:
            print(f"  [warn] failed parsing {path}: {e}")

    if not pairs:
        print("External QA parse completed with 0 usable QA pairs.")
        return []

    pairs.sort(key=_question_quality_score, reverse=True)
    selected = pairs[:max(0, int(max_questions))]
    print(f"Loaded external QA pairs: {len(selected)} selected from {len(pairs)} candidates")
    return selected


def build_instruction_text_dataset(qa_pairs, max_items=200, answer_override=None):
    rows = []
    for qa in qa_pairs[:max(0, int(max_items))]:
        prompt = _clean_qa_text(qa.get("prompt"))
        answer = _clean_qa_text(answer_override if answer_override is not None else qa.get("answer"))
        if not prompt or not answer:
            continue
        rows.append(f"Question: {prompt}\nAnswer: {answer}")
    if not rows:
        return Dataset.from_dict({"text": []})
    return Dataset.from_dict({"text": rows})


def ensure_text_dataset(dataset):
    if "text" in dataset.column_names:
        return dataset
    text_col = dataset.column_names[0]
    return dataset.map(
        lambda x: {"text": x[text_col]},
        remove_columns=dataset.column_names,
        num_proc=2,
    )


def concatenate_text_datasets(parts):
    usable = [ensure_text_dataset(p) for p in parts if p is not None and len(p) > 0]
    if not usable:
        return Dataset.from_dict({"text": []})
    if len(usable) == 1:
        return usable[0]
    return concatenate_datasets(usable)

def prepare_raw_text_continuation(dataset, num_samples=50):
    pairs, text_col = [], ('text' if 'text' in dataset.column_names else dataset.column_names[0])
    for item in dataset.select(range(min(num_samples, len(dataset)))):
        words = item[text_col].split()
        if len(words) >= 50:
            pairs.append({'prompt': " ".join(words[:30]), 'answer': " ".join(words[30:50])})
    return pairs


def apply_refusal_calibration(
    model,
    tokenizer,
    forget_questions,
    retain_questions,
    output_dir,
    refusal_text,
    epochs=1,
    max_forget=256,
    max_retain=128,
):
    forget_ds = build_instruction_text_dataset(
        forget_questions,
        max_items=max_forget,
        answer_override=refusal_text,
    )
    retain_ds = build_instruction_text_dataset(
        retain_questions,
        max_items=max_retain,
        answer_override=None,
    )

    calibration_ds = concatenate_text_datasets([forget_ds, retain_ds])
    if len(calibration_ds) == 0:
        print("Refusal calibration skipped: no calibration rows.")
        return model

    print(
        "Running refusal calibration | "
        f"rows={len(calibration_ds)} | forget={len(forget_ds)} | retain={len(retain_ds)} | epochs={epochs}"
    )
    model = fine_tune_model(
        model,
        tokenizer,
        calibration_ds,
        output_dir=output_dir,
        epochs=epochs,
        is_qa=True,
        use_discriminative_lr=True,
        lr_decay=0.90,
    )
    return model


def _get_model_input_device(model):
    """Best-effort device selection for generation/eval inputs."""
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight.device
    except Exception:
        pass
    return next(model.parameters()).device


def _ensure_set_submodule_compat():
    """Backport torch.nn.Module.set_submodule for older torch builds."""
    if hasattr(torch.nn.Module, "set_submodule"):
        return

    def _set_submodule(self, target, module):
        if not isinstance(target, str) or not target:
            raise ValueError("target must be a non-empty module path")
        atoms = target.split(".")
        parent = self
        for atom in atoms[:-1]:
            if atom.isdigit():
                parent = parent[int(atom)]
            else:
                parent = getattr(parent, atom)
        leaf = atoms[-1]
        if leaf.isdigit():
            parent[int(leaf)] = module
        else:
            setattr(parent, leaf, module)

    torch.nn.Module.set_submodule = _set_submodule


def _simple_overlap_f1(reference: str, prediction: str) -> float:
    """Fallback token-overlap score when rouge-score is unavailable."""
    ref_tokens = reference.lower().split()
    pred_tokens = prediction.lower().split()
    if not ref_tokens or not pred_tokens:
        return 0.0

    ref_counts = {}
    for tok in ref_tokens:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1

    pred_counts = {}
    for tok in pred_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1

    overlap = 0
    for tok, cnt in ref_counts.items():
        overlap += min(cnt, pred_counts.get(tok, 0))

    if overlap == 0:
        return 0.0
    precision = overlap / max(1, len(pred_tokens))
    recall = overlap / max(1, len(ref_tokens))
    return (2 * precision * recall) / max(1e-12, precision + recall)


def evaluate_rouge_score(model, tokenizer, qa_pairs, max_new_tokens=64):
    """Average ROUGE-L F1 over prompt/answer pairs."""
    if not qa_pairs:
        return float("nan")

    scorer = None
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    except Exception:
        scorer = None

    device = _get_model_input_device(model)
    scores = []
    was_training = model.training
    model.eval()

    with torch.no_grad():
        for pair in qa_pairs:
            prompt = str(pair.get("prompt", "")).strip()
            answer = str(pair.get("answer", "")).strip()
            if not prompt or not answer:
                continue

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQ_LEN,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=(tokenizer.pad_token_id or tokenizer.eos_token_id),
            )
            gen_text = tokenizer.decode(
                out_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()

            if scorer is not None:
                score = scorer.score(answer, gen_text)["rougeL"].fmeasure
            else:
                score = _simple_overlap_f1(answer, gen_text)
            scores.append(float(score))

    if was_training:
        model.train()
    return float(sum(scores) / len(scores)) if scores else float("nan")


def _extract_text_samples(dataset, num_samples=30):
    texts = []
    if hasattr(dataset, "column_names") and len(dataset) > 0:
        text_col = "text" if "text" in dataset.column_names else dataset.column_names[0]
        subset = dataset.select(range(min(num_samples, len(dataset))))
        for item in subset:
            text = str(item[text_col]).strip()
            if text:
                texts.append(text)
        return texts

    if isinstance(dataset, list):
        for item in dataset[:num_samples]:
            if isinstance(item, dict):
                text = str(item.get("text", item.get("prompt", ""))).strip()
            else:
                text = str(item).strip()
            if text:
                texts.append(text)
    return texts


def evaluate_perplexity(model, tokenizer, dataset, num_samples=30):
    """Average token-level perplexity over a small sample."""
    texts = _extract_text_samples(dataset, num_samples=num_samples)
    if not texts:
        return float("nan")

    device = _get_model_input_device(model)
    ppls = []
    was_training = model.training
    model.eval()

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQ_LEN,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = out.loss.detach().float()
            if torch.isfinite(loss):
                ppls.append(float(torch.exp(torch.clamp(loss, max=20)).item()))

    if was_training:
        model.train()
    return float(sum(ppls) / len(ppls)) if ppls else float("nan")


# ===== CELL 7 =====

# ── AMP compat shim (CPU-safe) ───────────────────────────────────────────────
def _amp_device():
    """Returns the device string for AMP — 'cuda' if available, 'cpu' otherwise."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def _make_scaler():
    """Returns a GradScaler for the current device. CPU scaler is a no-op.
    Falls back to cuda.amp for older PyTorch (<2.3).
    """
    if not torch.cuda.is_available():
        # CPU: return a dummy scaler whose methods are all no-ops
        class _NoOpScaler:
            def scale(self, loss): return loss
            def unscale_(self, opt): pass
            def step(self, opt): opt.step()
            def update(self): pass
        return _NoOpScaler()
    try: return torch.amp.GradScaler('cuda')
    except TypeError: return torch.cuda.amp.GradScaler()


# ============================================================
# Shock-Reduction Unlearning Methods
# ============================================================
import torch
import torch.nn.functional as F


# ── Manual SWA (Stochastic Weight Averaging) ─────────────────────────────────
class ManualSWA:
    """
    Lightweight SWA that works with device_map="auto" (model parallelism).
    Accumulates a running mean of parameter tensors on CPU in FP32 for
    numerical stability, then copies back to the model at the end.

    Averaging smooths out the erratic weight trajectory caused by gradient
    ascent, landing the model at a flatter, more stable "forgetting basin"
    rather than an extreme point that also hurts retain performance.
    """
    def __init__(self, model):
        self.n   = 0
        self.avg = {n: p.detach().cpu().float().clone()
                    for n, p in model.named_parameters()}

    def update(self, model):
        """Call this every swa_freq steps in the SWA phase."""
        self.n += 1
        with torch.no_grad():
            for name, param in model.named_parameters():
                cpu_param = param.detach().cpu().float()
                self.avg[name].add_((cpu_param - self.avg[name]) / self.n)

    def apply(self, model):
        """Copy averaged weights back into the model. Returns model."""
        print(f"  SWA: applying average of {self.n} snapshots...")
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(self.avg[name].to(param.device).to(param.dtype))
        return model


# ── Method 1: GA with inverted-triangle LR + SWA ────────────────────────────
def ga_layerwise_lr_unlearning(
    model, forget_dataloader,
    epochs: int = 3,
    base_lr: float = 1e-5,
    decay: float = 0.85,
    grad_accum: int = 4,
    max_grad_norm: float = 1.0,
    use_swa: bool = True,
    swa_start_frac: float = 0.5,  # start collecting after this fraction of total steps
    swa_freq: int = 5,
):
    """GA with per-layer discriminative LR. SWA averages the final checkpoints
    to prevent the model landing at an extreme point on the ascent trajectory."""
    print("\n[Method] GA — inverted-triangle LR" + (" + SWA" if use_swa else ""))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    param_groups = get_layerwise_param_groups(model, base_lr, decay)
    optimizer    = torch.optim.AdamW(param_groups)
    scaler       = _make_scaler()
    device       = next(model.parameters()).device

    total_steps  = epochs * len(forget_dataloader)
    swa_start    = int(total_steps * swa_start_frac)
    swa          = ManualSWA(model) if use_swa else None
    global_step  = 0

    for epoch in range(epochs):
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(forget_dataloader):
            inputs = batch["input_ids"].to(device)
            masks  = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(_amp_device()):
                out  = model(input_ids=inputs, attention_mask=masks, labels=labels)
                loss = -out.loss / grad_accum

            if not (torch.isnan(loss) or torch.isinf(loss)):
                scaler.scale(loss).backward()
            else:
                print(f"  [skip] step {step}: NaN/Inf")

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # SWA collection
                if use_swa and global_step >= swa_start and global_step % swa_freq == 0:
                    swa.update(model)

            total_loss += out.loss.item()
            global_step += 1

            if step % 20 == 0:
                print(f"  epoch {epoch} | step {step} | fwd-loss {out.loss.item():.4f}"
                      + (f" | swa_n={swa.n}" if use_swa and swa.n > 0 else ""))
                torch.cuda.empty_cache()

        print(f"  epoch {epoch} | avg fwd-loss {total_loss/max(1,len(forget_dataloader)):.4f}")

    if use_swa and swa.n > 0:
        swa.apply(model)
    return model


# ── Method 2: GA with gradual unfreezing + SWA ───────────────────────────────
def _freeze_all(model):
    for p in model.parameters(): p.requires_grad = False

def _unfreeze_last_n(model, n: int):
    try:
        layers = model.model.layers
    except AttributeError:
        for p in model.parameters(): p.requires_grad = True; return
    for layer in layers[-n:]:
        for p in layer.parameters(): p.requires_grad = True
    for p in model.lm_head.parameters(): p.requires_grad = True


def ga_gradual_unfreeze_unlearning(
    model, forget_dataloader,
    epochs: int = 6,
    lr: float = 1e-5,
    start_layers: int = 4,
    unfreeze_every_epoch: int = 2,
    grad_accum: int = 4,
    max_grad_norm: float = 1.0,
    use_swa: bool = True,
    swa_start_frac: float = 0.5,
    swa_freq: int = 5,
):
    """Gradual unfreezing: start with top layers, open more per phase.
    SWA on the final (fully unfrozen) phase."""
    print("\n[Method] GA — gradual unfreezing" + (" + SWA" if use_swa else ""))
    scaler = _make_scaler()
    device = next(model.parameters()).device

    try:
        num_layers = len(model.model.layers)
    except AttributeError:
        num_layers = 1

    current_unfrozen = 0
    total_steps      = epochs * len(forget_dataloader)
    swa_start        = int(total_steps * swa_start_frac)
    swa              = ManualSWA(model) if use_swa else None
    global_step      = 0

    for epoch in range(epochs):
        target = min(start_layers * (1 + epoch // unfreeze_every_epoch), num_layers)
        if target != current_unfrozen:
            _freeze_all(model)
            _unfreeze_last_n(model, target)
            current_unfrozen = target
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  epoch {epoch}: unfreezing last {target}/{num_layers} layers "
                  f"({trainable/1e6:.1f}M trainable params)")
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        model.train()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for step, batch in enumerate(forget_dataloader):
            inputs = batch["input_ids"].to(device)
            masks  = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(_amp_device()):
                out  = model(input_ids=inputs, attention_mask=masks, labels=labels)
                loss = -out.loss / grad_accum

            if not (torch.isnan(loss) or torch.isinf(loss)):
                scaler.scale(loss).backward()
            else:
                print(f"  [skip] step {step}: NaN/Inf")

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if use_swa and global_step >= swa_start and global_step % swa_freq == 0:
                    swa.update(model)

            total_loss += out.loss.item()
            global_step += 1

            if step % 20 == 0:
                print(f"  epoch {epoch} | step {step} | fwd-loss {out.loss.item():.4f}"
                      + (f" | swa_n={swa.n}" if use_swa and swa.n > 0 else ""))
                torch.cuda.empty_cache()

        print(f"  epoch {epoch} | avg fwd-loss {total_loss/max(1,len(forget_dataloader)):.4f}")

    if use_swa and swa.n > 0:
        swa.apply(model)
    for p in model.parameters(): p.requires_grad = True
    return model


# ── Method 3: GradDiff + SWA ─────────────────────────────────────────────────
def graddiff_unlearning(
    model, forget_dataloader, retain_dataloader,
    epochs: int = 3,
    lr: float = 1e-5,
    retain_weight: float = 1.0,
    grad_accum: int = 4,
    max_grad_norm: float = 1.0,
    use_swa: bool = True,
    swa_start_frac: float = 0.5,
    swa_freq: int = 5,
):
    """loss = -CE(forget) + retain_weight*CE(retain).
    Sequential forget/retain passes keep peak activation memory = 1 batch.
    SWA smooths the final checkpoint."""
    print("\n[Method] Gradient Difference" + (" + SWA" if use_swa else ""))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    optimizer    = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler       = _make_scaler()
    device       = next(model.parameters()).device
    retain_iter  = iter(retain_dataloader)

    total_steps  = epochs * len(forget_dataloader)
    swa_start    = int(total_steps * swa_start_frac)
    swa          = ManualSWA(model) if use_swa else None
    global_step  = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(epochs):
        total_fgt, total_ret = 0.0, 0.0

        for step, forget_batch in enumerate(forget_dataloader):
            # forget pass
            f_in  = forget_batch["input_ids"].to(device)
            f_msk = forget_batch["attention_mask"].to(device)
            f_lbl = forget_batch["labels"].to(device)
            with torch.amp.autocast(_amp_device()):
                f_out  = model(input_ids=f_in, attention_mask=f_msk, labels=f_lbl)
                f_loss = -f_out.loss / grad_accum
            if not (torch.isnan(f_loss) or torch.isinf(f_loss)):
                scaler.scale(f_loss).backward()

            # retain pass
            try:
                retain_batch = next(retain_iter)
            except StopIteration:
                retain_iter  = iter(retain_dataloader)
                retain_batch = next(retain_iter)
            r_in  = retain_batch["input_ids"].to(device)
            r_msk = retain_batch["attention_mask"].to(device)
            r_lbl = retain_batch["labels"].to(device)
            with torch.amp.autocast(_amp_device()):
                r_out  = model(input_ids=r_in, attention_mask=r_msk, labels=r_lbl)
                r_loss = retain_weight * r_out.loss / grad_accum
            if not (torch.isnan(r_loss) or torch.isinf(r_loss)):
                scaler.scale(r_loss).backward()

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if use_swa and global_step >= swa_start and global_step % swa_freq == 0:
                    swa.update(model)

            total_fgt += f_out.loss.item()
            total_ret += r_out.loss.item()
            global_step += 1

            if step % 20 == 0:
                print(f"  epoch {epoch} | step {step} | "
                      f"fgt {f_out.loss.item():.4f} | ret {r_out.loss.item():.4f}"
                      + (f" | swa_n={swa.n}" if use_swa and swa.n > 0 else ""))
                torch.cuda.empty_cache()

        n = max(1, len(forget_dataloader))
        print(f"  epoch {epoch} | avg fgt {total_fgt/n:.4f} | avg ret {total_ret/n:.4f}")

    if use_swa and swa.n > 0:
        swa.apply(model)
    return model

# ===== CELL 8 =====

# ── AMP hotfix: safe scaler for FP16-parameter training ─────────────────────
import inspect

def _make_scaler(model=None):
    """Return a scaler compatible with this notebook's mixed-precision setup.

    PyTorch GradScaler.unscale_ raises ValueError when gradients are FP16,
    which happens when trainable weights are loaded directly as torch.float16.
    In that case we intentionally fall back to a no-op scaler.
    """
    class _NoOpScaler:
        def scale(self, loss):
            return loss

        def unscale_(self, opt):
            pass

        def step(self, opt):
            opt.step()

        def update(self):
            pass

    if not torch.cuda.is_available():
        return _NoOpScaler()

    if model is None:
        # Infer model from caller (all training loops in this notebook expose `model`).
        frame = inspect.currentframe()
        try:
            caller = frame.f_back if frame is not None else None
            candidate = caller.f_locals.get("model") if caller is not None else None
            if isinstance(candidate, torch.nn.Module):
                model = candidate
        finally:
            del frame

    if model is None:
        # Conservative fallback for this notebook: avoid scaler/unscale crashes.
        return _NoOpScaler()

    has_fp16_trainable = any(
        p.requires_grad and p.dtype == torch.float16 for p in model.parameters()
    )
    if has_fp16_trainable:
        return _NoOpScaler()

    try:
        return torch.amp.GradScaler("cuda")
    except TypeError:
        return torch.cuda.amp.GradScaler()

# ===== CELL 9 =====

# ============================================================
# Enhanced Unlearning: NPO + GradDiff + Gradual Unfreeze
#                    + Inverted-Triangle LR + SWA
# ============================================================
#
# Algorithm overview
# ------------------
# We divide training into PHASES, where each phase gradually
# opens more transformer layers for gradient updates (gradual
# unfreezing). Within each phase:
#
#   loss = NPO(forget) + retain_weight * CE(retain)
#          ^^^^^^^^^^    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#          negative       GradDiff regularisation
#          preference     keeps retained knowledge intact
#          optimization
#
# Parameter groups are built with an inverted-triangle (layerwise)
# LR schedule: later (task-specific) layers get a higher LR, earlier
# (general) layers get a lower LR. This avoids destroying low-level
# linguistic representations while aggressively unlearning top-layer
# Harry-Potter-specific features.
#
# SWA runs during the final fraction of steps, averaging weight
# snapshots to smooth the noisy gradient-ascent trajectory and
# land in a flatter basin (better forget-retain trade-off).
# ============================================================

def _get_unfrozen_param_groups(model, base_lr: float, decay: float,
                                unfrozen_layer_ids: set):
    """
    Build layerwise LR groups for a subset of transformer layers.
    Layers not in unfrozen_layer_ids are frozen (requires_grad=False).
    Handles Gemma-3 tied weights (embed_tokens == lm_head) via seen_ids.
    """
    try:
        transformer_layers = model.model.layers
    except AttributeError:
        params = [p for p in model.parameters() if p.requires_grad]
        return [{"params": params, "lr": base_lr}]

    num_layers = len(transformer_layers)
    seen_ids   = set()
    groups     = []

    def _add(params, lr):
        unique = [p for p in params if id(p) not in seen_ids and p.requires_grad]
        seen_ids.update(id(p) for p in unique)
        if unique:
            groups.append({"params": unique, "lr": lr})

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad_(False)

    # Unfreeze & assign LR to selected layers
    for idx, layer in enumerate(transformer_layers):
        if idx in unfrozen_layer_ids:
            for p in layer.parameters():
                p.requires_grad_(True)
            _add(list(layer.parameters()),
                 base_lr * (decay ** (num_layers - 1 - idx)))

    # Always unfreeze lm_head with highest LR
    for p in model.lm_head.parameters():
        p.requires_grad_(True)
    _add(list(model.lm_head.parameters()), base_lr)

    active = sum(len(g["params"]) for g in groups)
    print(f"  Unfrozen layers: {sorted(unfrozen_layer_ids)} + lm_head "
          f"| {active} param tensors | LR range "
          f"[{min(g['lr'] for g in groups):.2e}, {max(g['lr'] for g in groups):.2e}]")
    return groups


def enhanced_unlearning(
    model,
    ref_model,
    forget_dataloader,
    retain_dataloader,
    # -- Gradual unfreezing schedule --
    total_phases:      int   = 3,     # number of unfreeze phases
    epochs_per_phase:  int   = 2,     # training epochs within each phase
    start_layers:      int   = 6,     # layers open in phase 0
    unfreeze_per_phase:int   = 4,     # additional layers opened each phase
    # -- Optimiser --
    base_lr:           float = 1e-5,
    lr_decay:          float = 0.85,
    grad_accum:        int   = 4,
    max_grad_norm:     float = 1.0,
    # -- NPO --
    beta:              float = 0.1,   # NPO temperature
    # -- GradDiff --
    retain_weight:     float = 1.0,   # weight on retain CE term
    # -- SWA --
    use_swa:           bool  = True,
    swa_start_frac:    float = 0.5,   # fraction of total steps after which SWA collects
    swa_freq:          int   = 5,
    max_update_steps   = None,
):
    """
    Combine NPO + GradDiff + Gradual Unfreezing + Inverted-Triangle LR + SWA
    into a single training loop.
    """
    num_layers = len(model.model.layers)
    total_epochs = total_phases * epochs_per_phase
    steps_per_epoch = len(forget_dataloader)
    total_steps = total_epochs * steps_per_epoch
    if max_update_steps is not None:
        total_steps = min(total_steps, int(max_update_steps))
    swa_start_step = int(total_steps * swa_start_frac)

    print(f"Enhanced Unlearning | phases={total_phases} | "
          f"epochs/phase={epochs_per_phase} | total_epochs={total_epochs}")
    print(f"Model layers={num_layers} | SWA starts at step {swa_start_step}/{total_steps}")

    model.train()
    ref_model.eval()

    # Derive ref_device from where ref_model actually lives.
    # - In real training: ref is loaded with device_map={"":0} → cuda:0.
    # - In pre-flight (MockGemma on CPU): ref stays on cpu.
    # Hardcoding "cuda:0" would break the pre-flight test when ref is CPU-only.
    ref_device    = next(ref_model.parameters()).device
    policy_device = next(model.parameters()).device

    scaler = _make_scaler()
    swa    = ManualSWA(model) if use_swa else None

    global_step    = 0
    retain_iter    = iter(retain_dataloader)
    stop_training  = False

    for phase in range(total_phases):
        # --- Compute the set of unfrozen layer indices for this phase ---
        # Start from the top (last) layers and progressively open earlier ones.
        top_layer = num_layers - 1
        n_open = start_layers + phase * unfreeze_per_phase
        n_open = min(n_open, num_layers)
        unfrozen_ids = set(range(top_layer - n_open + 1, top_layer + 1))

        print(f"\n── Phase {phase+1}/{total_phases}: "
              f"unfreezing {n_open} top layers ({min(unfrozen_ids)}-{max(unfrozen_ids)}) ──")

        param_groups = _get_unfrozen_param_groups(model, base_lr, lr_decay, unfrozen_ids)
        try:
            import bitsandbytes as bnb
            if not any(p.is_cuda for g in param_groups for p in g['params']):
                raise RuntimeError('bnb requires CUDA params')
            optimizer = bnb.optim.PagedAdamW8bit(param_groups)
        except Exception:
            optimizer = torch.optim.AdamW(param_groups)

        for epoch in range(epochs_per_phase):
            epoch_npo, epoch_retain, epoch_total = 0.0, 0.0, 0.0

            for step, forget_batch in enumerate(forget_dataloader):
                f_ids  = forget_batch["input_ids"].to(policy_device)
                f_msk  = forget_batch["attention_mask"].to(policy_device)
                f_lbl  = forget_batch["labels"].to(policy_device)

                # --- Retain batch (cycle if exhausted) ---
                try:
                    retain_batch = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_dataloader)
                    retain_batch = next(retain_iter)
                r_ids = retain_batch["input_ids"].to(policy_device)
                r_msk = retain_batch["attention_mask"].to(policy_device)
                r_lbl = retain_batch["labels"].to(policy_device)

                with torch.amp.autocast(_amp_device()):
                    # ── NPO forget loss ──────────────────────────────────────
                    outputs = model(input_ids=f_ids, attention_mask=f_msk)
                    policy_logps = get_batch_logps(outputs.logits, f_lbl)

                    with torch.no_grad():
                        ref_out = ref_model(
                            input_ids=f_ids.to(ref_device),
                            attention_mask=f_msk.to(ref_device),
                        )
                        ref_logps = get_batch_logps(
                            ref_out.logits,
                            f_lbl.to(ref_device),
                        ).to(policy_device)

                    ratio    = policy_logps - ref_logps
                    npo_loss = -(2.0 / beta) * F.logsigmoid(-beta * ratio).mean()

                    # ── GradDiff retain loss ─────────────────────────────────
                    r_out      = model(input_ids=r_ids, attention_mask=r_msk, labels=r_lbl)
                    retain_loss = r_out.loss

                    loss = (npo_loss + retain_weight * retain_loss) / grad_accum

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  [skip nan] phase={phase} epoch={epoch} step={step}")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                scaler.scale(loss).backward()

                if (step + 1) % grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in param_groups for p in g["params"]], max_grad_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    if use_swa and global_step >= swa_start_step and global_step % swa_freq == 0:
                        swa.update(model)

                    global_step += 1
                    if max_update_steps is not None and global_step >= int(max_update_steps):
                        print(f"  Reached ENH_MAX_STEPS={int(max_update_steps)}; stopping early.")
                        stop_training = True

                epoch_npo    += npo_loss.item()
                epoch_retain += retain_loss.item()
                epoch_total  += (npo_loss + retain_weight * retain_loss).item()

                if step % 20 == 0:
                    torch.cuda.empty_cache()
                    print(f"  p{phase+1} e{epoch} s{step} | "
                          f"npo={npo_loss.item():.4f} "
                          f"retain={retain_loss.item():.4f} "
                          f"total={epoch_total/(step+1):.4f}")

                if stop_training:
                    break

            n = len(forget_dataloader)
            print(f"  Phase {phase+1} Epoch {epoch} done | "
                  f"avg npo={epoch_npo/n:.4f} retain={epoch_retain/n:.4f}")
            if stop_training:
                break
        if stop_training:
            break

    if use_swa and swa and swa.n > 0:
        model = swa.apply(model)
    elif use_swa:
        print("  SWA: no snapshots collected (training too short). Returning final model.")

    # Restore all params to trainable
    for p in model.parameters():
        p.requires_grad_(True)

    return model


# ===== CELL 10 =====

# ============================================================
# Pre-flight Tests (CPU, no downloads, no GPU required)
# Runs in ~60 s on CPU before committing to Kaggle session.
# ============================================================
import traceback, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class _FakeLayer(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.linear = nn.Linear(d, d)
    def forward(self, x):
        return self.linear(x)

class MockGemma(nn.Module):
    """Tiny CPU model that mimics Gemma-3 structure for unit testing."""
    def __init__(self, vocab=256, d=32, n_layers=4):
        super().__init__()
        self.model = type("M", (), {
            "embed_tokens": nn.Embedding(vocab, d),
            "layers": nn.ModuleList([_FakeLayer(d) for _ in range(n_layers)])
        })()
        self.lm_head = nn.Linear(d, vocab, bias=False)
        # Tie weights (Gemma-3 does this)
        self.lm_head.weight = self.model.embed_tokens.weight
        self.config = type("C", (), {"use_cache": False})()

    def named_parameters(self, **kw):
        yield from super().named_parameters(**kw)

    def parameters(self, **kw):
        yield from super().parameters(**kw)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        class _Out:
            pass
        out = _Out()
        out.logits, out.loss = logits, loss
        return out

def _make_fake_batch(bs=2, seq=8, vocab=256):
    ids = torch.randint(1, vocab, (bs, seq))
    msk = torch.ones_like(ids)
    return {"input_ids": ids, "attention_mask": msk, "labels": ids.clone()}

PASS, FAIL = 0, 0

def run_test(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  [PASS] {name}")
        PASS += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        FAIL += 1

print("=" * 60)
print("Pre-flight Tests — Enhanced Unlearning Notebook")
print("=" * 60)

VOCAB, D, N_LAYERS = 256, 32, 4

# T1: ManualSWA instantiation + update + apply
def t_swa():
    m = MockGemma(VOCAB, D, N_LAYERS)
    swa = ManualSWA(m)
    swa.update(m)
    swa.apply(m)
    assert swa.n == 1
run_test("ManualSWA_init_update_apply", t_swa)

# T2: _get_unfrozen_param_groups — partial unfreeze
def t_partial_unfreeze():
    m = MockGemma(VOCAB, D, N_LAYERS)
    groups = _get_unfrozen_param_groups(m, base_lr=1e-4, decay=0.85,
                                        unfrozen_layer_ids={2, 3})
    assert len(groups) >= 1, "expected at least 1 param group"
    # All frozen params must have requires_grad=False except the unfrozen set
    frozen = [p for n, p in m.named_parameters()
              if not any(p is gp for g in groups for gp in g["params"])]
    assert all(not p.requires_grad for p in frozen), "frozen param has grad enabled"
run_test("partial_unfreezing_grad_flags", t_partial_unfreeze)

# T3: _get_unfrozen_param_groups — no duplicate params (tied weights)
def t_no_duplicates():
    m = MockGemma(VOCAB, D, N_LAYERS)
    groups = _get_unfrozen_param_groups(m, 1e-4, 0.85, {0,1,2,3})
    all_params = [p for g in groups for p in g["params"]]
    ids = [id(p) for p in all_params]
    assert len(ids) == len(set(ids)), "duplicate param tensors in groups"
run_test("no_duplicate_params_tied_weights", t_no_duplicates)

# T4: get_batch_logps shape
def t_batch_logps():
    m = MockGemma(VOCAB, D, N_LAYERS)
    ids = torch.randint(1, VOCAB, (2, 8))
    labels = ids.clone(); labels[0, :2] = -100
    with torch.no_grad():
        out = m(input_ids=ids, labels=labels)
        lp = get_batch_logps(out.logits, labels)
    assert lp.shape == (2,), f"shape mismatch {lp.shape}"
run_test("get_batch_logps_shape", t_batch_logps)

# T5: NPO loss is finite and scalar
def t_npo_finite():
    m   = MockGemma(VOCAB, D, N_LAYERS)
    ref = MockGemma(VOCAB, D, N_LAYERS)
    ref.eval()
    batch = _make_fake_batch()
    with torch.no_grad():
        ref_out  = ref(**batch)
        ref_logps = get_batch_logps(ref_out.logits, batch["labels"])
    m_out = m(**batch)
    policy_logps = get_batch_logps(m_out.logits, batch["labels"])
    ratio = policy_logps - ref_logps
    npo = -(2.0/0.1) * F.logsigmoid(-0.1 * ratio).mean()
    assert torch.isfinite(npo), f"NPO loss not finite: {npo}"
run_test("npo_loss_finite", t_npo_finite)

# T6: GradDiff retain loss is finite
def t_graddiff_retain():
    m = MockGemma(VOCAB, D, N_LAYERS)
    batch = _make_fake_batch()
    out = m(**batch)
    assert torch.isfinite(out.loss), "retain CE loss not finite"
run_test("graddiff_retain_loss_finite", t_graddiff_retain)

# T7: enhanced_unlearning runs 1 phase / 1 epoch (smoke test)
def t_enhanced_smoke():
    m   = MockGemma(VOCAB, D, N_LAYERS)
    ref = MockGemma(VOCAB, D, N_LAYERS)
    from torch.utils.data import DataLoader, TensorDataset
    ids = torch.randint(1, VOCAB, (4, 8))
    ds  = TensorDataset(ids, torch.ones_like(ids), ids.clone())
    class _DL:
        """Minimal DataLoader-like that yields dict batches."""
        def __init__(self, ids):
            self.ids = ids
        def __iter__(self):
            for i in range(0, len(self.ids), 2):
                b = self.ids[i:i+2]
                yield {"input_ids": b, "attention_mask": torch.ones_like(b), "labels": b.clone()}
        def __len__(self): return len(self.ids) // 2
    fl = _DL(ids)
    rl = _DL(ids)
    result = enhanced_unlearning(
        m, ref, fl, rl,
        total_phases=1, epochs_per_phase=1,
        start_layers=2, unfreeze_per_phase=0,
        grad_accum=1, use_swa=True, swa_start_frac=0.0,
    )
    assert result is not None
run_test("enhanced_unlearning_smoke", t_enhanced_smoke)

# T8: CPU _make_scaler returns a no-op when CUDA unavailable
def t_cpu_scaler():
    scaler = _make_scaler()
    if not torch.cuda.is_available():
        loss = torch.tensor(1.0, requires_grad=True)
        scaled = scaler.scale(loss)
        assert scaled is loss, "NoOpScaler.scale should return loss unchanged"
run_test("cpu_noop_scaler", t_cpu_scaler)

# T9: _amp_device returns correct string
def t_amp_device():
    d = _amp_device()
    assert d in ("cuda", "cpu")
    if not torch.cuda.is_available():
        assert d == "cpu"
run_test("amp_device_string", t_amp_device)

# T10: tokenize_dataset works on tiny dataset (skips if no network)
def t_dataloader():
    from datasets import Dataset
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained("gpt2", timeout=5)
    except Exception as e:
        print(f"    (skipping — tokenizer download failed: {e})")
        return
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = Dataset.from_dict({"text": ["hello world " * 20, "foo bar " * 20]})
    td = tokenize_dataset(ds, tok, max_length=16)
    assert len(td) > 0, "tokenized dataset is empty"
run_test("tokenize_dataset_basic", t_dataloader)

print("=" * 60)
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL:
    print("FIX FAILURES BEFORE RUNNING ON KAGGLE")
else:
    print("All tests passed — safe to run on Kaggle.")
print("=" * 60)


# ===== CELL 11 =====

import os, json, shutil, gc
import torch, pandas as pd
from transformers import AutoModelForCausalLM, AutoConfig
import math


def _env_bool(name, default):
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name, default):
    v = os.getenv(name)
    return int(v) if v is not None else default


def _env_float(name, default):
    v = os.getenv(name)
    return float(v) if v is not None else default

os.makedirs(MODELS_DIR, exist_ok=True)

EXPERIMENT_TAG = os.getenv("ENH_EXPERIMENT", "default").strip() or "default"
_default_ckpt = CHECKPOINTS["ENHANCED"]
if EXPERIMENT_TAG == "default":
    enhanced_ckpt = os.getenv("ENHANCED_CKPT_PATH", _default_ckpt)
else:
    enhanced_ckpt = os.getenv("ENHANCED_CKPT_PATH", f"{_default_ckpt}_{EXPERIMENT_TAG}")

hp_total_phases = _env_int("ENH_TOTAL_PHASES", 3)
hp_epochs_per_phase = _env_int("ENH_EPOCHS_PER_PHASE", 2)
hp_start_layers = _env_int("ENH_START_LAYERS", 6)
hp_unfreeze_per_phase = _env_int("ENH_UNFREEZE_PER_PHASE", 4)
hp_base_lr = _env_float("ENH_BASE_LR", 1e-5)
hp_lr_decay = _env_float("ENH_LR_DECAY", 0.85)
hp_grad_accum = _env_int("ENH_GRAD_ACCUM", 4)
hp_beta = _env_float("ENH_BETA", 0.1)
hp_retain_weight = _env_float("ENH_RETAIN_WEIGHT", 1.0)
hp_use_swa = _env_bool("ENH_USE_SWA", True)
hp_swa_start_frac = _env_float("ENH_SWA_START_FRAC", 0.5)
hp_swa_freq = _env_int("ENH_SWA_FREQ", 5)
hp_max_steps_raw = _env_int("ENH_MAX_STEPS", -1)
hp_max_steps = hp_max_steps_raw if hp_max_steps_raw > 0 else None
hp_train_forget_qa_max = _env_int("ENH_TRAIN_FORGET_QA_MAX", 240)
hp_train_retain_qa_max = _env_int("ENH_TRAIN_RETAIN_QA_MAX", 240)

flag_use_external_qa = _env_bool("ENH_USE_EXTERNAL_QA", True)
external_qa_path = os.getenv("ENH_EXTERNAL_QA_PATH", "/home/bhaskar/inlp/data/harrypotterqa")
external_qa_max = _env_int("ENH_EXTERNAL_QA_MAX", 300)

flag_refusal_calibration = _env_bool("ENH_REFUSAL_CALIBRATION", True)
hp_refusal_epochs = _env_int("ENH_REFUSAL_EPOCHS", 1)
hp_refusal_max_forget = _env_int("ENH_REFUSAL_MAX_FORGET", 256)
hp_refusal_max_retain = _env_int("ENH_REFUSAL_MAX_RETAIN", 128)
hp_refusal_text = os.getenv(
    "ENH_REFUSAL_TEXT",
    "I can't help with Harry Potter specific details.",
).strip()

flag_retrain_enhanced = _env_bool("RETRAIN_ENHANCED", FLAGS["RETRAIN_ENHANCED"])
flag_requantize = _env_bool("REQUANTIZE", FLAGS["REQUANTIZE"])
flag_skip_full_eval = _env_bool("SKIP_FULL_EVAL", False)

print(
    f"Experiment={EXPERIMENT_TAG} | ckpt={enhanced_ckpt} | "
    f"retrain={flag_retrain_enhanced} | requantize={flag_requantize} | "
    f"skip_full_eval={flag_skip_full_eval}"
)
print(
    "HP: "
    f"phases={hp_total_phases}, epochs/phase={hp_epochs_per_phase}, "
    f"start_layers={hp_start_layers}, unfreeze/phase={hp_unfreeze_per_phase}, "
    f"base_lr={hp_base_lr}, lr_decay={hp_lr_decay}, grad_accum={hp_grad_accum}, "
    f"beta={hp_beta}, retain_weight={hp_retain_weight}, "
    f"use_swa={hp_use_swa}, swa_start_frac={hp_swa_start_frac}, swa_freq={hp_swa_freq}, "
    f"max_steps={hp_max_steps}, train_forget_qa_max={hp_train_forget_qa_max}, "
    f"train_retain_qa_max={hp_train_retain_qa_max}, use_external_qa={flag_use_external_qa}, "
    f"external_qa_max={external_qa_max}, external_qa_path={external_qa_path}, "
    f"refusal_calibration={flag_refusal_calibration}"
)

# ── Resources ─────────────────────────────────────────────────────────────────
print("=== Loading resources ===")
forget_train, retain_train, forget_qa, retain_qa = load_muse_data()
tokenizer = load_tokenizer_only()
forget_train_eval = ensure_text_dataset(forget_train)
retain_train_eval = ensure_text_dataset(retain_train)
base_forget_train_n = len(forget_train_eval)
base_retain_train_n = len(retain_train_eval)

forget_qa_questions   = prepare_cloze_questions(forget_qa.select(range(min(50, len(forget_qa)))))
retain_qa_questions   = prepare_cloze_questions(retain_qa.select(range(min(50, len(retain_qa)))))
manual_forget_questions, manual_retain_questions = get_actual_eval_questions()
forget_qa_questions = merge_eval_questions(manual_forget_questions, forget_qa_questions)
retain_qa_questions = merge_eval_questions(manual_retain_questions, retain_qa_questions)

if flag_use_external_qa:
    external_forget_questions = load_external_qa_pairs(external_qa_path, max_questions=external_qa_max)
    if external_forget_questions:
        forget_qa_questions = merge_eval_questions(forget_qa_questions, external_forget_questions)
        print(f"Merged external forget QA pairs: +{len(external_forget_questions)} candidates")

forget_train_aug = build_instruction_text_dataset(
    forget_qa_questions,
    max_items=hp_train_forget_qa_max,
)
retain_train_aug = build_instruction_text_dataset(
    retain_qa_questions,
    max_items=hp_train_retain_qa_max,
)

forget_train = concatenate_text_datasets([forget_train, forget_train_aug])
retain_train = concatenate_text_datasets([retain_train, retain_train_aug])

forget_raw_pairs      = prepare_raw_text_continuation(forget_train_eval, num_samples=50)
retain_raw_pairs      = prepare_raw_text_continuation(retain_train_eval, num_samples=50)
print(f"QA:  {len(forget_qa_questions)} forget / {len(retain_qa_questions)} retain")
print(f"Manual QA seeds included: {len(manual_forget_questions)} forget + {len(manual_retain_questions)} retain")
print(
    "Train signal expanded: "
    f"forget {base_forget_train_n}->{len(forget_train)} | "
    f"retain {base_retain_train_n}->{len(retain_train)}"
)
print(f"Raw: {len(forget_raw_pairs)} forget / {len(retain_raw_pairs)} retain")
clean_memory()

# ── Enhanced unlearning ───────────────────────────────────────────────────────
print("\n=== Enhanced Unlearning (NPO + GradDiff + GradualUnfreeze + LayerwiseLR + SWA) ===")
if not os.path.exists(enhanced_ckpt) or flag_retrain_enhanced:
    policy_model, _ = load_base_model(gradient_checkpointing=True)
    # Pin ref_model to cuda:0 only — avoids tied-weight device mismatch
    # when device_map="auto" shards lm_head (tied to embed_tokens on cuda:0)
    # across devices, causing RuntimeError on the lm_head forward pass.
    from transformers import AutoModelForCausalLM as _AMCLM
    print("Loading reference model (pinned to cuda:0)...")
    _ref_device_map = {"": 0} if torch.cuda.is_available() else "cpu"
    ref_model = _AMCLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=_ref_device_map,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    ref_model.config.use_cache = False
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    forget_loader = create_dataloader(forget_train, tokenizer, batch_size=2)
    retain_loader = create_dataloader(retain_train, tokenizer, batch_size=2)

    policy_model = enhanced_unlearning(
        model            = policy_model,
        ref_model        = ref_model,
        forget_dataloader= forget_loader,
        retain_dataloader= retain_loader,
        total_phases      = hp_total_phases,
        epochs_per_phase  = hp_epochs_per_phase,
        start_layers      = hp_start_layers,
        unfreeze_per_phase= hp_unfreeze_per_phase,
        base_lr           = hp_base_lr,
        lr_decay          = hp_lr_decay,
        grad_accum        = hp_grad_accum,
        beta              = hp_beta,
        retain_weight     = hp_retain_weight,
        use_swa           = hp_use_swa,
        swa_start_frac    = hp_swa_start_frac,
        swa_freq          = hp_swa_freq,
        max_update_steps  = hp_max_steps,
    )

    if flag_refusal_calibration:
        calib_dir = os.path.join(MODELS_DIR, f"refusal_calibration_{EXPERIMENT_TAG}")
        policy_model = apply_refusal_calibration(
            model=policy_model,
            tokenizer=tokenizer,
            forget_questions=forget_qa_questions,
            retain_questions=retain_qa_questions,
            output_dir=calib_dir,
            refusal_text=hp_refusal_text,
            epochs=hp_refusal_epochs,
            max_forget=hp_refusal_max_forget,
            max_retain=hp_refusal_max_retain,
        )

    print(f"Saving enhanced model → {enhanced_ckpt}...")
    policy_model.save_pretrained(enhanced_ckpt)
    tokenizer.save_pretrained(enhanced_ckpt)
    AutoConfig.from_pretrained(MODEL_NAME).save_pretrained(enhanced_ckpt)

    del ref_model, forget_loader, retain_loader
    clean_memory()
else:
    print(f"Checkpoint found: {enhanced_ckpt}")
    policy_model = None

if flag_skip_full_eval:
    if EXPERIMENT_TAG == "default":
        results_csv = os.getenv("RESULTS_CSV", "enhanced_unlearning_results.csv")
    else:
        results_csv = os.getenv("RESULTS_CSV", f"enhanced_unlearning_results_{EXPERIMENT_TAG}.csv")

    quick_df = pd.DataFrame([
        {
            "Model": "Enhanced (FP16)",
            "Checkpoint": enhanced_ckpt,
            "Skipped_Full_Eval": True,
            "Experiment": EXPERIMENT_TAG,
        }
    ])
    quick_df.to_csv(results_csv, index=False)
    print(f"SKIP_FULL_EVAL=1 -> saved quick marker CSV -> {results_csv}")

    import wandb
    wandb.finish()
    raise SystemExit(0)

# ── Evaluation ────────────────────────────────────────────────────────────────
print("\n=== Evaluation ===")
results = []

def run_evaluation(model_or_path, label, is_quantized=False):
    print(f"  Evaluating {label}...")
    try:
        own_model = isinstance(model_or_path, str)
        if own_model:
            if is_quantized:
                from transformers import BitsAndBytesConfig
                _ensure_set_submodule_compat()
                bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
                m = AutoModelForCausalLM.from_pretrained(
                    model_or_path, quantization_config=bnb, device_map="auto",
                    config=AutoConfig.from_pretrained(MODEL_NAME))
            else:
                m = AutoModelForCausalLM.from_pretrained(
                    model_or_path, torch_dtype=torch.float16, device_map="auto",
                    attn_implementation="eager")
        else:
            m = model_or_path

        r = {
            "Model":           label,
            "Forget QA ROUGE": evaluate_rouge_score(m, tokenizer, forget_qa_questions),
            "Retain QA ROUGE": evaluate_rouge_score(m, tokenizer, retain_qa_questions),
            "Forget Raw ROUGE":evaluate_rouge_score(m, tokenizer, forget_raw_pairs),
            "Retain Raw ROUGE":evaluate_rouge_score(m, tokenizer, retain_raw_pairs),
            "Forget PPL":      evaluate_perplexity(m, tokenizer, forget_train_eval,  num_samples=30),
            "Retain PPL":      evaluate_perplexity(m, tokenizer, retain_train_eval,  num_samples=30),
        }
        if own_model:
            del m; clean_memory()
        return r
    except Exception as e:
        traceback.print_exc()
        nan = float("nan")
        return {"Model": label,
                "Forget QA ROUGE": nan, "Retain QA ROUGE": nan,
                "Forget Raw ROUGE": nan, "Retain Raw ROUGE": nan,
                "Forget PPL": nan, "Retain PPL": nan, "Error": str(e)}

# Base model baseline
print("  Loading base model for baseline eval...")
base_model, _ = load_base_model()
results.append(run_evaluation(base_model, "Base (FP16)"))
del base_model; clean_memory()

# Enhanced model — use in-memory if just trained, else load from checkpoint
if policy_model is not None:
    results.append(run_evaluation(policy_model, "Enhanced (in-memory)"))
    del policy_model; clean_memory()

if os.path.exists(enhanced_ckpt):
    results.append(run_evaluation(enhanced_ckpt, "Enhanced (FP16)", is_quantized=False))
    if flag_requantize:
        results.append(run_evaluation(enhanced_ckpt, "Enhanced (4-bit)", is_quantized=True))

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Results ===")
df = pd.DataFrame(results)
print(df.to_string(index=False))
if EXPERIMENT_TAG == "default":
    results_csv = os.getenv("RESULTS_CSV", "enhanced_unlearning_results.csv")
else:
    results_csv = os.getenv("RESULTS_CSV", f"enhanced_unlearning_results_{EXPERIMENT_TAG}.csv")
df.to_csv(results_csv, index=False)
print(f"Saved results CSV -> {results_csv}")

failed = [r["Model"] for r in results
          if any(isinstance(v, float) and math.isnan(v) for v in r.values() if isinstance(v, (int, float)))]
if failed:
    print(f"WARNING: NaN metrics for: {failed}")
else:
    print("All evaluations produced valid metrics.")

import wandb
wandb.finish()
