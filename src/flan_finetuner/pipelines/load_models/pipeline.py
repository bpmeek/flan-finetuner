"""
This is a boilerplate pipeline 'load_models'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import load_original_model, load_tokenizer, tokenize_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_original_model,
                inputs="params:model_name",
                outputs="original_model",
                name="load_original_model",
            ),
            node(
                func=load_tokenizer,
                inputs=["params:model_name", "params:model_path"],
                outputs="tokenizer",
                name="load_tokenizer",
            ),
            node(
                func=tokenize_dataset,
                inputs=["tokenizer", "dataset"],
                outputs="tokenized_datasets",
                name="tokenize_datasets",
            ),
        ]
    )
