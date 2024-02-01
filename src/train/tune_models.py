import os
import joblib
import pickle
import optuna
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler

from src.utils import get_project_path, PATH_TO_DATA, PATH_TO_MODELS
from src.train.objectives import catboost_objective, mlp_objective, rf_objective


def tune_all_estimators():

    x_train = np.load(os.path.join(PATH_TO_DATA, 'processed', 'x_train.npy'), allow_pickle=True)
    x_test = np.load(os.path.join(PATH_TO_DATA, 'processed', 'x_test.npy'), allow_pickle=True)
    y_train = np.load(os.path.join(PATH_TO_DATA, 'processed', 'y_train.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(PATH_TO_DATA, 'processed', 'y_test.npy'), allow_pickle=True)

    sc = StandardScaler()
    norm = Normalizer()
    min_max = MinMaxScaler()

    sc.fit(x_train)
    norm.fit(x_train)
    min_max.fit(x_train)

    save_path_process = os.path.join(PATH_TO_MODELS, 'process')

    pickle.dump(sc, open(os.path.join(save_path_process, 'StandardScaler.pkl'), 'wb'))
    pickle.dump(norm, open(os.path.join(save_path_process, 'Normalizer.pkl'), 'wb'))
    pickle.dump(min_max, open(os.path.join(save_path_process, 'MinMaxScaler.pkl'), 'wb'))

    for i, name in enumerate(['pure_data', 'sc_data', 'norm_data', 'min_max_data']):
        if name == 'pure_data':
            x_train_tmp = x_train
            x_test_tmp = x_test
        if name == 'sc_data':
            x_train_tmp = sc.transform(x_train)
            x_test_tmp = sc.transform(x_test)
        if name == 'norm_data':
            x_train_tmp = norm.transform(x_train)
            x_test_tmp = norm.transform(x_test)
        if name == 'min_max_data':
            x_train_tmp = min_max.transform(x_train)
            x_test_tmp = min_max.transform(x_test)

        for estimator in ['MLP', 'catboost', 'random_forest']:

            study = optuna.create_study(
                study_name=f'photocatalysis_{estimator}_{name}',
                directions=['minimize', 'minimize', 'maximize'],
                pruner=optuna.pruners.MedianPruner(),
                load_if_exists=True,
                storage=f'sqlite:///{get_project_path()}/optuna_models_optimization.db'
            )

            if estimator == 'MLP':
                study.optimize(lambda trial: mlp_objective(trial, X_train=x_train_tmp,
                                                           y_train=y_train, X_test=x_test_tmp,
                                                           y_test=y_test), n_trials=500)

            if estimator == 'catboost':
                study.optimize(lambda trial: catboost_objective(trial, X_train=x_train_tmp,
                                                                y_train=y_train, X_test=x_test_tmp,
                                                                y_test=y_test), n_trials=500)

            if estimator == 'random_forest':
                study.optimize(lambda trial: rf_objective(trial, X_train=x_train_tmp,
                                                          y_train=y_train, X_test=x_test_tmp,
                                                          y_test=y_test), n_trials=500)

            if not os.path.exists(os.path.join(PATH_TO_MODELS, estimator, name)):
                os.makedirs(os.path.join(PATH_TO_MODELS, estimator, name))

            joblib.dump(study, os.path.join(get_project_path(), 'models', estimator, name, 'study.pkl'))


if __name__ == "__main__":
    tune_all_estimators()