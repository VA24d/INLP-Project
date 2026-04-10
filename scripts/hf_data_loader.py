#!/usr/bin/env python3
"""Reusable Hugging Face model and dataset loader utilities.

This script can be imported from other modules, or run directly as a CLI:

python scripts/hf_data_loader.py \
  --model-name google/gemma-3-1b-it \
  --dataset-name muse-bench/MUSE-Books \
  --dataset-config raw \
  --split forget \
  --text-column text \
  --batch-size 2 \
  --max-length 512 \
  --load-model
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

TEXT_COLUMN_CANDIDATES = ("text", "content", "prompt", "input", "question")


@dataclass
class ModelLoadOptions:
    load_model_weights: bool = False
    load_in_4bit: bool = False
    dtype: Optional[torch.dtype] = None
    device_map: str = "auto"
    use_fast_tokenizer: bool = True
    trust_remote_code: bool = False


@dataclass
class DataLoaderOptions:
    text_column: Optional[str] = None
    batch_size: int = 2
    max_length: int = 512
    max_samples: Optional[int] = None
    num_workers: int = 0
    shuffle: bool = True


def resolve_dtype(dtype_name: str) -> Optional[torch.dtype]:
    dtype_name = dtype_name.lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name == "auto":
        return None
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def resolve_hf_token(explicit_token: Optional[str], token_env: str) -> Optional[str]:
    if explicit_token:
        return explicit_token
    token = os.environ.get(token_env)
    return token if token else None


def load_tokenizer(
    model_name_or_path: str,
    *,
    token: Optional[str] = None,
    use_fast: bool = True,
    add_pad_token: bool = True,
    trust_remote_code: bool = False,
) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=use_fast,
        token=token,
        trust_remote_code=trust_remote_code,
    )
    if add_pad_token and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _make_4bit_config(load_in_4bit: bool) -> Optional[BitsAndBytesConfig]:
    if not load_in_4bit:
        return None
    if not torch.cuda.is_available():
        print(
            "Warning: --load-in-4bit requested but CUDA is not available. "
            "Proceeding without 4-bit quantization."
        )
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def load_model(
    model_name_or_path: str,
    *,
    token: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    device_map: str = "auto",
    load_in_4bit: bool = False,
    trust_remote_code: bool = False,
) -> PreTrainedModel:
    quant_config = _make_4bit_config(load_in_4bit)
    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        token=token,
        torch_dtype=dtype,
        device_map=device_map,
        quantization_config=quant_config,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )


def load_hf_split(
    dataset_name: str,
    *,
    split: str,
    dataset_config: Optional[str] = None,
    token: Optional[str] = None,
) -> Dataset:
    try:
        return load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            token=token,
        )
    except Exception as exc:
        try:
            ds_dict = load_dataset(dataset_name, dataset_config, token=token)
            if isinstance(ds_dict, DatasetDict):
                available = ", ".join(ds_dict.keys())
            else:
                available = "unknown"
        except Exception:
            available = "unknown"
        raise ValueError(
            f"Could not load split '{split}' from '{dataset_name}'. "
            f"Available splits: {available}"
        ) from exc


def infer_text_column(dataset: Dataset, requested: Optional[str] = None) -> str:
    if requested:
        if requested not in dataset.column_names:
            raise ValueError(
                f"Requested text column '{requested}' not found. "
                f"Columns: {dataset.column_names}"
            )
        return requested

    for col in TEXT_COLUMN_CANDIDATES:
        if col in dataset.column_names:
            return col

    first_row = dataset[0]
    for col in dataset.column_names:
        if isinstance(first_row.get(col), str):
            return col

    raise ValueError(
        "Could not infer a text column automatically. "
        f"Columns: {dataset.column_names}"
    )


def tokenize_for_causal_lm(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    *,
    text_column: str,
    max_length: int,
    num_proc: Optional[int] = None,
) -> Dataset:
    def _tokenize(batch):
        return tokenizer(
            batch[text_column],
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(
        _tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc=f"Tokenizing '{text_column}'",
    )


def build_causal_lm_dataloader(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    *,
    text_column: Optional[str] = None,
    max_length: int = 512,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 0,
    max_samples: Optional[int] = None,
    num_proc: Optional[int] = None,
) -> Tuple[DataLoader, str]:
    resolved_text_column = infer_text_column(dataset, text_column)

    if max_samples is not None:
        sample_count = min(max_samples, len(dataset))
        dataset = dataset.select(range(sample_count))

    tokenized = tokenize_for_causal_lm(
        dataset,
        tokenizer,
        text_column=resolved_text_column,
        max_length=max_length,
        num_proc=num_proc,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataloader, resolved_text_column


def build_loader_bundle(
    *,
    model_name_or_path: str,
    dataset_name: str,
    split: str,
    dataset_config: Optional[str] = None,
    token: Optional[str] = None,
    model_options: Optional[ModelLoadOptions] = None,
    dataloader_options: Optional[DataLoaderOptions] = None,
) -> Tuple[Optional[PreTrainedModel], PreTrainedTokenizerBase, Dataset, DataLoader, str]:
    if model_options is None:
        model_options = ModelLoadOptions()
    if dataloader_options is None:
        dataloader_options = DataLoaderOptions()

    tokenizer = load_tokenizer(
        model_name_or_path,
        token=token,
        use_fast=model_options.use_fast_tokenizer,
        add_pad_token=True,
        trust_remote_code=model_options.trust_remote_code,
    )

    model: Optional[PreTrainedModel] = None
    if model_options.load_model_weights:
        model = load_model(
            model_name_or_path,
            token=token,
            dtype=model_options.dtype,
            device_map=model_options.device_map,
            load_in_4bit=model_options.load_in_4bit,
            trust_remote_code=model_options.trust_remote_code,
        )

    dataset = load_hf_split(
        dataset_name,
        split=split,
        dataset_config=dataset_config,
        token=token,
    )
    dataloader, resolved_text_column = build_causal_lm_dataloader(
        dataset,
        tokenizer,
        text_column=dataloader_options.text_column,
        max_length=dataloader_options.max_length,
        batch_size=dataloader_options.batch_size,
        shuffle=dataloader_options.shuffle,
        num_workers=dataloader_options.num_workers,
        max_samples=dataloader_options.max_samples,
    )
    return model, tokenizer, dataset, dataloader, resolved_text_column


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load Hugging Face model/tokenizer + dataset split and build a PyTorch DataLoader."
    )
    parser.add_argument("--model-name", required=True, help="HF model id or local model path")
    parser.add_argument("--dataset-name", required=True, help="HF dataset id or local dataset path")
    parser.add_argument("--dataset-config", default=None, help="Dataset config/subset name")
    parser.add_argument("--split", default="train", help="Dataset split (e.g., train, forget, retain1)")
    parser.add_argument("--text-column", default=None, help="Text column name. If omitted, inferred.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-shuffle", action="store_true", help="Disable shuffling")

    parser.add_argument("--load-model", action="store_true", help="Also load model weights")
    parser.add_argument("--load-in-4bit", action="store_true", help="Use 4-bit loading when CUDA is available")
    parser.add_argument("--dtype", default="auto", help="auto|float16|bfloat16|float32")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map value")
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--hf-token", default=None, help="Hugging Face token (optional)")
    parser.add_argument(
        "--hf-token-env",
        default="HF_TOKEN",
        help="Environment variable to read token from when --hf-token is not set",
    )
    parser.add_argument("--slow-tokenizer", action="store_true", help="Disable fast tokenizer")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    token = resolve_hf_token(args.hf_token, args.hf_token_env)
    dtype = resolve_dtype(args.dtype)

    model_options = ModelLoadOptions(
        load_model_weights=args.load_model,
        load_in_4bit=args.load_in_4bit,
        dtype=dtype,
        device_map=args.device_map,
        use_fast_tokenizer=not args.slow_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    dataloader_options = DataLoaderOptions(
        text_column=args.text_column,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        shuffle=not args.no_shuffle,
    )

    model, tokenizer, dataset, dataloader, text_column = build_loader_bundle(
        model_name_or_path=args.model_name,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        token=token,
        model_options=model_options,
        dataloader_options=dataloader_options,
    )

    batch = next(iter(dataloader))

    print("\n=== Hugging Face Loader Summary ===")
    print(f"Model Name/Path: {args.model_name}")
    print(f"Dataset Name:    {args.dataset_name}")
    print(f"Dataset Config:  {args.dataset_config}")
    print(f"Split:           {args.split}")
    print(f"Text Column:     {text_column}")
    print(f"Rows:            {len(dataset)}")
    print(f"Batch Size:      {args.batch_size}")
    print(f"input_ids shape: {tuple(batch['input_ids'].shape)}")
    print(f"labels shape:    {tuple(batch['labels'].shape)}")

    preview = tokenizer.decode(batch["input_ids"][0], skip_special_tokens=False)
    preview = preview[:300].replace("\n", " ")
    print(f"Preview sample:  {preview}")

    if model is not None:
        print(f"Model loaded:    {type(model).__name__}")


if __name__ == "__main__":
    main()
