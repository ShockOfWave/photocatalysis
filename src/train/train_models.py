import os
import optuna
import pickle
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.progress import Progress
from typing import List
from itertools import product
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

from src.utils import get_project_path, PATH_TO_DATA, PATH_TO_MODELS, PATH_TO_REPORTS


def prepare_data_and_params(name: str, estimator: str) -> List:
    x_train, x_test, y_train, y_test = load_data()

    if name == 'pure_data':
        x_train_tmp = x_train
        x_test_tmp = x_test
    else:
        process = load_process(name)
        x_train_tmp = process.transform(x_train)
        x_test_tmp = process.transform(x_test)

    params = get_params_of_best_model(name=name, estimator=estimator)

    return [params, x_train_tmp, x_test_tmp, y_train, y_test]


def load_data() -> List:
    path_to_data = os.path.join(PATH_TO_DATA, 'processed')
    x_train = np.load(os.path.join(path_to_data, 'x_train.npy'), allow_pickle=True)
    x_test = np.load(os.path.join(path_to_data, 'x_test.npy'), allow_pickle=True)
    y_train = np.load(os.path.join(path_to_data, 'y_train.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(path_to_data, 'y_test.npy'), allow_pickle=True)
    return [x_train, x_test, y_train, y_test]


def get_params_of_best_model(name: str, estimator: str) -> dict:
    study_name = f'photocatalysis_{estimator}_{name}'
    storage = f'sqlite:///{os.path.join(get_project_path(), "optuna_models_optimization.db")}'
    study = optuna.load_study(study_name=study_name, storage=storage)
    return study.best_trials[0].params


def load_process(process: str):

    if process == 'sc_data':
        process = pickle.load(open(os.path.join(PATH_TO_MODELS, 'process', 'StandardScaler.pkl'), 'rb'))
    elif process == 'norm_data':
        process = pickle.load(open(os.path.join(PATH_TO_MODELS, 'process', 'Normalizer.pkl'), 'rb'))
    else:
        process = pickle.load(open(os.path.join(PATH_TO_MODELS, 'process', 'MinMaxScaler.pkl'), 'rb'))

    return process


def train_catboost(name: str, estimator: str) -> List:
    params, x_train_tmp, x_test_tmp, y_train, y_test = prepare_data_and_params(name=name, estimator=estimator)

    model = CatBoostRegressor(**params)

    model.fit(x_train_tmp, y_train, silent=True)

    return [model, x_train_tmp, x_test_tmp, y_train, y_test, params]


def train_mlp(name: str, estimator: str) -> List:
    params, x_train_tmp, x_test_tmp, y_train, y_test = prepare_data_and_params(name=name, estimator=estimator)

    layers = []

    hidden_layer_sizes = []

    for key in list(params.keys()):
        if key.startswith('n_units'):
            layers.append(key)
            hidden_layer_sizes.append(params[key])

    hidden_layer_sizes = tuple(hidden_layer_sizes)

    layers.append('n_layers')

    for key in layers:
        params.pop(key, None)

    params['hidden_layer_sizes'] = hidden_layer_sizes

    model = MLPRegressor(**params)

    model.fit(x_train_tmp, y_train)

    return [model, x_train_tmp, x_test_tmp, y_train, y_test, params]


def train_rf(name: str, estimator: str) -> List:
    params, x_train_tmp, x_test_tmp, y_train, y_test = prepare_data_and_params(name=name, estimator=estimator)

    model = RandomForestRegressor(**params)

    model.fit(x_train_tmp, y_train)

    return [model, x_train_tmp, x_test_tmp, y_train, y_test, params]


def save_results(model, x_train_tmp, x_test_tmp, y_train, y_test, name: str, estimator: str, params: dict):
    savepath_figs = os.path.join(PATH_TO_REPORTS, 'figures', estimator, name)
    savepath_tables = os.path.join(PATH_TO_REPORTS, 'tables', estimator, name)
    savepath_models = os.path.join(PATH_TO_MODELS, estimator, name)

    if not os.path.exists(savepath_figs):
        os.makedirs(savepath_figs)

    if not os.path.exists(savepath_tables):
        os.makedirs(savepath_tables)

    if not os.path.exists(savepath_models):
        os.makedirs(savepath_models)

    preds_train = model.predict(x_train_tmp)
    preds_test = model.predict(x_test_tmp)

    mae_train = mean_absolute_error(y_train, preds_train)
    mse_train = mean_squared_error(y_train, preds_train)
    r2_train = r2_score(y_train, y_train)

    mae_test = mean_absolute_error(y_test, preds_test)
    mse_test = mean_squared_error(y_test, preds_test)
    r2_test = r2_score(y_test, preds_test)

    metrics = {
        'Mean absolute error train': mae_train,
        'Mean squared error train': mse_train,
        'R2 score train': r2_train,
        'Mean absolute error test': mae_test,
        'Mean squared error test': mse_test,
        'R2 score test': r2_test,
    }

    fig, ax = plt.subplots()
    ax.plot(y_train, y_train, 'red', label='True values VS true values')
    ax.scatter(y_train, preds_train, s=15, label='True values VS predicted values')
    ax.set_title('Regression plot for train data', fontsize=20)
    ax.set_xlabel('True values', fontsize=20)
    # ax.set_xticks(fontsize=15)
    ax.set_ylabel('Predicted values', fontsize=20)
    # ax.set_yticks(fontsize=15)
    ax.legend(loc='lower right', fontsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)

    text = '\n'.join((
        r'$MAE=%.5f$' % (mae_train, ),
        r'$MSE=%.5f$' % (mse_train, ),
        r'$R^2=%.5f$' % (r2_train, )
    ))

    ax.text(0.05, 0.95, text, fontsize=15, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    fig.tight_layout()

    fig.savefig(os.path.join(savepath_figs, 'train_reg_plot.svg'), format='svg', dpi=1200)
    fig.savefig(os.path.join(savepath_figs, 'train_reg_plot.png'), format='png', dpi=1200)
    fig.savefig(os.path.join(savepath_figs, 'train_reg_plot.pdf'), format='pdf', dpi=1200)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(y_test, y_test, 'red', label='True values VS true values')
    ax.scatter(y_test, preds_test, s=15, label='True values VS predicted values')
    ax.set_title('Regression plot for test data', fontsize=20)
    ax.set_xlabel('True values', fontsize=20)
    # ax.set_xticks(fontsize=15)
    ax.set_ylabel('Predicted values', fontsize=20)
    # ax.set_yticks(fontsize=15)
    ax.legend(loc='lower right', fontsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)

    text = '\n'.join((
        r'$MAE=%.5f$' % (mae_test,),
        r'$MSE=%.5f$' % (mse_test,),
        r'$R^2=%.5f$' % (r2_test,)
    ))

    ax.text(0.05, 0.95, text, fontsize=15, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    fig.tight_layout()

    fig.savefig(os.path.join(savepath_figs, 'test_reg_plot.svg'), format='svg', dpi=1200)
    fig.savefig(os.path.join(savepath_figs, 'test_reg_plot.png'), format='png', dpi=1200)
    fig.savefig(os.path.join(savepath_figs, 'test_reg_plot.pdf'), format='pdf', dpi=1200)
    plt.close(fig)

    pd.DataFrame.from_dict([metrics]).to_csv(os.path.join(savepath_tables, 'metrics.csv'))
    pd.DataFrame.from_dict([params]).to_csv(os.path.join(savepath_tables, 'best_params.csv'))

    pickle.dump(model, open(os.path.join(savepath_models, 'model.pkl'), 'wb'))


def plot_feature_imp(estimator: str, name: str, model, x_train):
    features_names = pd.read_csv(os.path.join(PATH_TO_DATA, 'raw', 'data.csv'), delimiter=';').columns.tolist()
    features_names.pop(0)
    features_names.pop(-1)
    savepath = os.path.join(PATH_TO_REPORTS, 'figures', estimator, name)
    if estimator == 'catboost' or estimator == 'random_forest':
        explaner = shap.TreeExplainer(model, feature_names=features_names)
    else:
        x_train_summary = shap.kmeans(x_train, len(features_names))
        explaner = shap.KernelExplainer(model.predict, x_train_summary, feature_names=features_names)

    shap_values = explaner(x_train)

    shap.summary_plot(shap_values, x_train, show=False)

    plt.tight_layout()
    plt.savefig(os.path.join(savepath, 'beeswarm.svg'), format='svg', dpi=1200)
    plt.savefig(os.path.join(savepath, 'beeswarm.png'), format='png', dpi=1200)
    plt.savefig(os.path.join(savepath, 'beeswarm.pdf'), format='pdf', dpi=1200)
    plt.close()


def train_models():
    names = ['pure_data', 'sc_data', 'norm_data', 'min_max_data']
    estimators = ['MLP', 'catboost', 'random_forest']

    with Progress() as progress:
        catboost_task = progress.add_task("[yellow]Training CatBoost...", total=len(names))
        mlp_task = progress.add_task('[blue]Training MLP...', total=len(names))
        rf_task = progress.add_task('[green]Training Random Forest...', total=len(names))

        for name, estimator in product(names, estimators):
            if estimator == 'catboost':
                model, x_train_tmp, x_test_tmp, y_train, y_test, params = train_catboost(name=name, estimator=estimator)
                save_results(model, x_train_tmp, x_test_tmp, y_train, y_test, name, estimator, params)
                plot_feature_imp(estimator, name, model, x_train_tmp)
                progress.update(catboost_task, advance=1)
            elif estimator == 'MLP':
                model, x_train_tmp, x_test_tmp, y_train, y_test, params = train_mlp(name=name, estimator=estimator)
                save_results(model, x_train_tmp, x_test_tmp, y_train, y_test, name, estimator, params)
                plot_feature_imp(estimator, name, model, x_train_tmp)
                progress.update(mlp_task, advance=1)
            else:
                model, x_train_tmp, x_test_tmp, y_train, y_test, params = train_rf(name=name, estimator=estimator)
                save_results(model, x_train_tmp, x_test_tmp, y_train, y_test, name, estimator, params)
                plot_feature_imp(estimator, name, model, x_train_tmp)
                progress.update(rf_task, advance=1)
