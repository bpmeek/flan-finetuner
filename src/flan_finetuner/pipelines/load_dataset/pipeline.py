"""
This is a boilerplate pipeline 'load_dataset'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import huggingface_load, process_dataset, dataframes_to_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=huggingface_load,
                inputs="params:dataset_name",
                outputs="raw_dataset",
                name="load_huggingface_dataset",
            ),
            node(
                func=process_dataset,
                inputs=["raw_dataset", "params:train"],
                outputs="train_df",
                name="process_train",
            ),
            node(
                func=process_dataset,
                inputs=["raw_dataset", "params:test"],
                outputs="test_df",
                name="process_test",
            ),
            node(
                func=dataframes_to_dataset,
                inputs=["train_df", "test_df"],
                outputs="dataset",
                name="dataframes_to_dataset",
            ),
        ]
    )
