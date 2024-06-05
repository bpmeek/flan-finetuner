"""
This is a boilerplate pipeline 'train'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import build_peft_model, get_training_args, get_peft_trainer, train_and_save


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=build_peft_model,
                inputs=["original_model", "params:lora_config"],
                outputs="peft_model",
                name="build_peft_model",
            ),
            node(
                func=get_training_args,
                inputs="params:training_arguments",
                outputs="training_args",
                name="get_training_args",
            ),
            node(
                func=get_peft_trainer,
                inputs=["peft_model", "training_args", "tokenized_datasets"],
                outputs="peft_trainer",
                name="get_peft_trainer",
            ),
            node(
                func=train_and_save,
                inputs=["peft_trainer", "params:model_path"],
                outputs="trained_model",
                name="train_and_save",
            ),
        ]
    )
