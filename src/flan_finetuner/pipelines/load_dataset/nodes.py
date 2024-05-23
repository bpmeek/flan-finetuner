"""
This is a boilerplate pipeline 'load_dataset'
generated using Kedro 0.19.5
"""
from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd


def huggingface_load(dataset_name: str) -> DatasetDict:
    return load_dataset(dataset_name)


def process_dataset(datasets: DatasetDict, dataset: str) -> pd.DataFrame:
    df = pd.DataFrame(data=datasets[dataset])

    df = df.explode('messages')
    df["raw_message"] = df.apply(lambda x: str(x["messages"].get("content")), axis=1)
    df["role"] = df.apply(lambda x: str(x["messages"].get("role")), axis=1)
    return df.loc[df["prompt"] != df["raw_message"]]


def dataframes_to_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame) -> DatasetDict:
    train = Dataset.from_pandas(train_df.reset_index(drop=True))
    test = Dataset.from_pandas(test_df.reset_index(drop=True))
    return DatasetDict({"train": train, "test": test})
