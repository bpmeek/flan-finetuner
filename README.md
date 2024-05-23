# flan-finetuner

## Overview

This project was created with Kedro, it is meant to serve as a repository of pipelines for fine-tuning LLM's, 
specifically `google/flan-t5-base`. 

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a [data engineering convention](https://docs.kedro.org/en/stable/faq/faq.html#what-is-data-engineering-convention)
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to install dependencies

Dependencies are declared in `requirements.txt` for `pip` installation.

To install them, run:

```
pip install -r requirements.txt
```

## How to run your Kedro pipeline

You will first have to update the `model_path` and `training_arguments.output_dir` to your own storage locations.

You can run your Kedro project with:

```
kedro run
```

