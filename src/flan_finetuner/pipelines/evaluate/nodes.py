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

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig


def update_model(base_model: AutoModelForSeq2SeqLM.from_pretrained, model_path: str):
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

    inputs = datasets['test'][0:10]['input_ids']

    human_baseline_summaries = datasets['test'][0:10]['labels']

    peft_summaries = []
    for idx, input_ids in enumerate(inputs):

        outputs = model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
        model_text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

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
