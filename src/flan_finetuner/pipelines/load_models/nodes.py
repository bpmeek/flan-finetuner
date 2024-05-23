"""
This is a boilerplate pipeline 'load_models'
generated using Kedro 0.19.5
"""
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch


def load_original_model(model_name) -> AutoModelForSeq2SeqLM:
    return AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='cuda')


def load_tokenizer(model_name: str, model_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(model_path)
    return tokenizer


def tokenize_dataset(tokenizer: AutoTokenizer.from_pretrained, dataset):

    def tokenize_function(example):
        prompt = ["Role: User\n\n" + dialogue + "\n\nRole: Assistant " for dialogue in example["prompt"]]
        example["input_ids"] = tokenizer(
            prompt, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids
        example["labels"] = tokenizer(
            example["raw_message"], padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids
        return example

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(
        ["prompt", "prompt_id", "messages", "category", "raw_message", "role"]
    )
    return tokenized_datasets
