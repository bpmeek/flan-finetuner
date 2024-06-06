"""
This is a boilerplate pipeline 'train'
generated using Kedro 0.19.5
"""
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

from typing import Dict, Any, Tuple


def build_peft_model(
    original_model: AutoModelForSeq2SeqLM.from_pretrained, lora_config: Dict[str, Any]
):
    lora_config["task_type"] = TaskType.SEQ_2_SEQ_LM
    lora_config = LoraConfig(**lora_config)
    return get_peft_model(original_model, lora_config)


def get_training_args(training_arguments: Dict[str, Any]):
    return TrainingArguments(**training_arguments)


def get_peft_trainer(peft_model, peft_training_args, tokenized_datasets) -> Trainer:
    tokenized_datasets = tokenized_datasets.filter(
        lambda example, index: index % 100 == 0, with_indices=True
    ).with_format("torch")
    return Trainer(
        model=peft_model,
        args=peft_training_args,
        train_dataset=tokenized_datasets["train"],
    )


def train_and_save(peft_trainer: Trainer, model_path: str) -> str:
    peft_trainer.train()
    peft_trainer.model.save_pretrained(model_path)
    return model_path
