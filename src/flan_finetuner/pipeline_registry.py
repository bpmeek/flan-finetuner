"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline, pipeline

from .pipelines import load_dataset, load_models, train, evaluate


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    load_dataset_pipeline = load_dataset.create_pipeline()
    load_dataset_pipeline = load_dataset_pipeline.to_outputs("raw_dataset")
    load_models_pipe = load_models.create_pipeline()
    load_models_pipeline = pipeline(
        pipe=load_models_pipe,
        inputs={"dataset": "raw_dataset"}
    )
    train_pipeline = train.create_pipeline()
    evaluation_pipeline = evaluate.create_pipeline()

    pipelines = find_pipelines()
    pipelines["__default__"] = load_dataset_pipeline + load_models_pipeline + train_pipeline + evaluation_pipeline
    pipelines["load"] = load_dataset.create_pipeline()
    pipelines["evaluation"] = load_dataset_pipeline + evaluation_pipeline
    return pipelines
