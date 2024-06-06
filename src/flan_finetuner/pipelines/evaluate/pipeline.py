"""
This is a boilerplate pipeline 'evaluate'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import update_model, generate_dialogue, rouge_evaluation


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=update_model,
            inputs=[
                "original_model",
                "params:training_arguments.output_dir"
            ],
            outputs="loaded_model",
            name="update_model"
        ),
        node(
            func=generate_dialogue,
            inputs=[
                "tokenized_datasets"
                "loaded_model",
                "tokenizer"
            ],
            outputs="dialogues",
            name="generate_dialogue"
        ),
        node(
            func=rouge_evaluation,
            inputs="dialogues",
            outputs="rogue_metric",
            name="rouge_evaluation"
        )
    ])
