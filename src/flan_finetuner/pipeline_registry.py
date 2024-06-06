"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from .pipelines import load_dataset, evaluate


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    load_pipeline = load_dataset.create_pipeline()
    evaluation_pipeline = evaluate.create_pipeline()

    pipelines = find_pipelines()
    pipelines["__default__"] = sum(pipelines.values())
    pipelines["load"] = load_dataset.create_pipeline()
    pipelines["evaluation"] = load_pipeline + evaluation_pipeline
    return pipelines
