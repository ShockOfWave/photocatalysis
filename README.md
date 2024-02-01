# Photocatalysis

![GitHub](https://img.shields.io/github/license/ShockOfWave/photocatalysis)
![GitHub last commit](https://img.shields.io/github/last-commit/ShockOfWave/photocatalysis)
![GitHub pull requests](https://img.shields.io/github/issues-pr/ShockOfWave/photocatalysis)
![contributors](https://img.shields.io/github/contributors/ShockOfWave/photocatalysis) 
![codesize](https://img.shields.io/github/languages/code-size/ShockOfWave/photocatalysis)
![GitHub repo size](https://img.shields.io/github/repo-size/ShockOfWave/photocatalysis)
![GitHub top language](https://img.shields.io/github/languages/top/ShockOfWave/photocatalysis)
![GitHub language count](https://img.shields.io/github/languages/count/ShockOfWave/photocatalysis)

# Introduction

In this study, a machine learning method was applied to determine the optimal conditions for the synthesis of graphitic carbon nitride  (g-C3N4) for use in photocatalytic hydrogen production processes. The main concept of this work is that predictive machine learning algorithms make it possible to develop a method for assessing the photocatalytic activity of a catalyst, which in turn reduces the cost of resources and effort compared to traditional computational and/or experimental methods. In connection with this task, a database was experimentally formed by obtaining g-C3N4 samples by heat treatment of nitrogen-containing precursors in air at a temperature of 450–600°C with varying synthesis heating times and rates.. Graphitic carbon nitride based materials were characterized by physicochemical analysis methods including X-ray phase analysis and low temperature nitrogen adsorption.Based on the obtained experimental data, a table for machine learning was generated. The results of the study showed that the use of machine learning can significantly improve the synthesis process of g-C3N4 material and increase its efficiency in the process of photocatalytic hydrogen evolution. This means that through the use of machine learning, more accurate predictions of the properties of graphitic carbon nitride depending on the parameters of its synthesis, which in turn can lead to the directed synthesis of a highly active material based on g-C3N4 used as catalysts in the reaction of photocatalytic hydrogen production.

# Project structure

- In [data folder](data/) you can find our dataset
- In [reports folder](reports/) you can find figures with metrics and models evaluations
- In [models folder](models/) you can find fitted models
- In [serialized models folder](serialized_models/) you can find fitted label encoders
- In [src folder](src/) you can find code
- In [application folder](app/) you can find code for frontend
- In [db](optuna_models_optimization.db) you can find hyperparameter optimization results

# Installation

## Clone repository

```
git clone git@github.com:ShockOfWave/photocatalysis.git
```

## Install dependencies with [Poetry](https://python-poetry.org)

```
poetry install
```

# Usage

## Training
You can run data processing, models tuning and models fitting with CLI:
```
poetry run python -m src --help
```

## Training results
You can check the results of hyperparameter optimization using the optuna-dashboard library
```
poetry run optuna-dashboard sqlite:///optuna_models_optimization.db
```

You can use your database or ours.

## CLI usage
The simple CLI supports multiple commands for data preparation, optimization, and model training. You can see all possible flags for it using the command:
```
poetry run python -m src --help
```

## Web UI usage
This app based on [streamlit](https://streamlit.io) library:
```bash
streamlit run app.py
```
Set login and password for Web UI in .streamlit/secrets.toml
```
[passwords]

user = "password"
```

## Run app with [docker](https://www.docker.com)
### Build docker image:
```
docker build -t shockofwave/photocatalysis:latest .
```

### Run app in docker:
```
docker run -d -p 8511:8511 --restart=always shockofwave/photocatalysis
```

# Acknowledgments/References
We thank the [Infochemistry Scientific Center ISC](https://infochemistry.ru) for the computing resouces.

We thank the [Boreskov Institute of Catalysis](https://catalysis.ru) for the provide data.

# Reference & Citation

Will be later...

# License
The code is distributed under the [MIT license](https://opensource.org/license/mit/).
