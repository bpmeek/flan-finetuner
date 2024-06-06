"""
This is a boilerplate pipeline 'evaluate'
generated using Kedro 0.19.5
"""
from typing import Dict

import pandas as pd

import evaluate
from datasets import Dataset
from peft import PeftModel

import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import logging

log = logging.getLogger(__name__)


def update_model(
        base_model: AutoModelForSeq2SeqLM.from_pretrained,
        model_path: str
):
    return PeftModel.from_pretrained(
        base_model,
        model_path,
        torch_dtype=torch.bfloat16,
        is_trainable=False
    )


def generate_dialogue(
        datasets: Dataset,
        model: PeftModel.from_pretrained,
        tokenizer: AutoTokenizer.from_pretrained,
) -> pd.DataFrame:
    datasets = datasets.with_format("torch")

    dialogues = datasets['test'][0:10]['dialogue']

    human_baseline_summaries = datasets['test'][0:10]['summary']

    peft_summaries = []
    for idx, dialogue in enumerate(dialogues):
        prompt = f"""
Summarize the following conversation.

{dialogue}

Summary: """

        input_ids = tokenizer(prompt, return_tensors='pt').input_ids

        outputs = model.generate(
            input_ids=input_ids
        )
        model_text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        log.info(f"Model output: {model_text_output}")
        log.info(f"Human baseline: {human_baseline_summaries[idx]}")

        peft_summaries.append(model_text_output)

    zipped_summaries = list(zip(human_baseline_summaries, peft_summaries))
    df = pd.DataFrame(zipped_summaries, columns=["human_baseline_summaries", "peft_summaries"])
    return df


def rouge_evaluation(summaries: pd.DataFrame) -> Dict[str, float]:
    rouge = evaluate.load('rouge')
    model_results = rouge.compute(
        predictions=summaries["peft_summaries"],
        references=summaries["human_baseline_summaries"],
        use_aggregator=True,
        use_stemmer=True,
    )
    return {"rouge_result": model_results}
