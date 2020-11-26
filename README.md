# 2020 SIOP Machine Learning Competition

## Overview

[Challenge overview](https://eval.ai/web/challenges/challenge-page/527/overview)

[FAQ](https://docs.google.com/document/d/1gxWAl5jMtZXabcOPd2ivTT-C3KAXA_E3BLHcF80dtXs/edit)

## Previous Years

[2019](https://github.com/izk8/2019_SIOP_Machine_Learning_Winners)

[2018](https://github.com/izk8/2018_SIOP_Machine_Learning_Winners)

# Setup

## Conda

Install the [conda](https://docs.conda.io/en/latest/miniconda.html#)
environment:

```
conda env create -f environment.yml
```

To activate the environment:

```
conda activate siop_ml_2020
```

To update the environment specify it in the `environment.yml` and then
([see _Updating an environment_](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#updating-an-environment)):

```
conda env update --file environment.yml  --prune
```

## Pre-commit

With the `siop_ml_2020` conda environment activated install the
[pre-commit](https://pre-commit.com/) Git hooks:

```
pre-commit install
```

## Data

[Description of the data](https://drive.google.com/open?id=1_Ve4jRoYsj5GB62_9BXdJGjstl_2KsZ5)

[Training data set](https://drive.google.com/open?id=1a0ltph5u7cD8TKEtprYDt2TpmW8On2b_)

[Development data set](https://drive.google.com/open?id=15ZHCnMuWYKrcXAE1ugJsJS7xG6krEQW-)

## Visual Studio Code

[Download](https://code.visualstudio.com/Download)

Make sure to install the recommended extensions listed under
`.vscode/extensions.json`. To see this in the UI find the
**Extensions: Show Recommended Extensions** command ([see _Workspace recommended extensions_](https://code.visualstudio.com/docs/editor/extension-gallery#_workspace-recommended-extensions))

To use the correct Python environment make sure to select the interperer
([see _Select and activate an environment_](https://code.visualstudio.com/docs/python/environments#_select-and-activate-an-environment)).

Visual Studio Code may run Python code interactively with the Jupyter
environment ([see _Working with Jupyter Notebooks in Visual Studio Code_](https://code.visualstudio.com/docs/python/jupyter-support)).
