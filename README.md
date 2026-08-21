# Photocatalysis

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.ijhydene.2024.07.245-blue)](https://doi.org/10.1016/j.ijhydene.2024.07.245)
[![CI](https://github.com/ShockOfWave/photocatalysis/actions/workflows/ci.yml/badge.svg)](https://github.com/ShockOfWave/photocatalysis/actions/workflows/ci.yml)
![GitHub](https://img.shields.io/github/license/ShockOfWave/photocatalysis)
![GitHub last commit](https://img.shields.io/github/last-commit/ShockOfWave/photocatalysis)
![GitHub pull requests](https://img.shields.io/github/issues-pr/ShockOfWave/photocatalysis)
![contributors](https://img.shields.io/github/contributors/ShockOfWave/photocatalysis)
![codesize](https://img.shields.io/github/languages/code-size/ShockOfWave/photocatalysis)
![GitHub repo size](https://img.shields.io/github/repo-size/ShockOfWave/photocatalysis)
![GitHub top language](https://img.shields.io/github/languages/top/ShockOfWave/photocatalysis)
![GitHub language count](https://img.shields.io/github/languages/count/ShockOfWave/photocatalysis)

This repository contains the code, the trained models and the hyperparameter optimization results behind the article published in *International Journal of Hydrogen Energy* — see [Reference & Citation](#reference--citation).

# Introduction

In this study, a machine learning method was applied to determine the optimal conditions for the synthesis of graphitic carbon nitride (g-C3N4) for use in photocatalytic hydrogen production processes. The main concept of this work is that predictive machine learning algorithms make it possible to develop a method for assessing the photocatalytic activity of a catalyst, which in turn reduces the cost of resources and effort compared to traditional computational and/or experimental methods. In connection with this task, a database was experimentally formed by obtaining g-C3N4 samples by heat treatment of nitrogen-containing precursors in air at a temperature of 450–600°C with varying synthesis heating times and rates. Graphitic carbon nitride based materials were characterized by physicochemical analysis methods including X-ray phase analysis and low temperature nitrogen adsorption. Based on the obtained experimental data, a table for machine learning was generated. The results of the study showed that the use of machine learning can significantly improve the synthesis process of g-C3N4 material and increase its efficiency in the process of photocatalytic hydrogen evolution. This means that through the use of machine learning, more accurate predictions of the properties of graphitic carbon nitride depending on the parameters of its synthesis, which in turn can lead to the directed synthesis of a highly active material based on g-C3N4 used as catalysts in the reaction of photocatalytic hydrogen production.

Three algorithms — gradient boosting (CatBoost), a multilayer perceptron and a random forest — were each optimized with [Optuna](https://optuna.org) over four preprocessing variants (no processing, `StandardScaler`, `Normalizer`, `MinMaxScaler`), giving twelve tuned models. Feature importances were analysed with [SHAP](https://github.com/shap/shap).

# Project structure

- In [data folder](data/) you can find our dataset
- In [reports folder](reports/) you can find figures and metrics of the model evaluations: [`reports/figures/`](reports/figures/) holds the regression plots and SHAP beeswarms per algorithm and preprocessing variant, [`reports/tables/`](reports/tables/) holds the corresponding metrics and best hyperparameters
- In [models folder](models/) you can find all twelve fitted models together with their Optuna studies, and the fitted scalers and label encoders in [`models/process/`](models/process/)
- In [serialized models folder](serialized_models/) you can find the model served by the web application, together with the `Normalizer` and the label encoders it needs
- In [src folder](src/) you can find code
- In [application folder](app/) you can find code for frontend
- In [db](optuna_models_optimization.db) you can find hyperparameter optimization results

# Installation

## Clone repository

```bash
git clone git@github.com:ShockOfWave/photocatalysis.git
```

## Install dependencies with [Poetry](https://python-poetry.org)

```bash
poetry install
```

To install only what the web application needs, skip the training extras:

```bash
poetry install --without train
```

# Usage

## Training
You can run data processing, models tuning and models fitting with CLI:
```bash
poetry run python -m src --help
```

## Training results
You can check the results of hyperparameter optimization using the optuna-dashboard library
```bash
poetry run optuna-dashboard sqlite:///optuna_models_optimization.db
```

You can use your database or ours.

## CLI usage
The simple CLI supports multiple commands for data preparation, optimization, and model training. You can see all possible flags for it using the command:
```bash
poetry run python -m src --help
```

## Web UI usage
This app is based on the [streamlit](https://streamlit.io) library:
```bash
poetry run streamlit run app.py
```
Set login and password for Web UI in .streamlit/secrets.toml
```
[passwords]

user = "password"
```

## Run app with [docker](https://www.docker.com)
### Build docker image:
```bash
docker build -t shockofwave/photocatalysis:latest .
```

### Run app in docker:
```bash
docker run -d -p 8511:8511 --restart=always shockofwave/photocatalysis
```

# Acknowledgments/References
We thank the [Infochemistry Scientific Center ISC](https://infochemistry.ru) for the computing resources.

We thank the [Boreskov Institute of Catalysis](https://catalysis.ru) for providing the data.

# Reference & Citation

Yurova, V.Yu., Potapenko, K.O., Aliev, T.A., Kozlova, E.A., Skorb, E.V. (2024) Optimization of g-C3N4 synthesis parameters based on machine learning to predict the efficiency of photocatalytic hydrogen production. *International Journal of Hydrogen Energy*, **81**, 193–203. https://doi.org/10.1016/j.ijhydene.2024.07.245

```bibtex
@article{yurova2024photocatalysis,
  title   = {Optimization of g-C3N4 synthesis parameters based on machine learning to predict the efficiency of photocatalytic hydrogen production},
  author  = {Yurova, Veronika Yu. and Potapenko, Kseniya O. and Aliev, Timur A. and Kozlova, Ekaterina A. and Skorb, Ekaterina V.},
  journal = {International Journal of Hydrogen Energy},
  volume  = {81},
  pages   = {193--203},
  year    = {2024},
  doi     = {10.1016/j.ijhydene.2024.07.245}
}
```

# License
The code is distributed under the [MIT license](https://opensource.org/license/mit/).
