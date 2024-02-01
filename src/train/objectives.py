from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor


def catboost_objective(trial, X_train, y_train, X_test, y_test):
    model = CatBoostRegressor(
        iterations=trial.suggest_int("iterations", 1000, 5000),
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        depth=trial.suggest_int("depth", 4, 10),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True),
        bootstrap_type=trial.suggest_categorical("bootstrap_type", ["Bayesian"]),
        random_strength=trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 10.0),
        od_type=trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
        od_wait=trial.suggest_int("od_wait", 10, 50),
        verbose=False,
        task_type='GPU',
        devices=[0, 1]
    )

    model.fit(X_train, y_train)

    return (mean_absolute_error(y_test, model.predict(X_test)),
            mean_squared_error(y_test, model.predict(X_test)),
            r2_score(y_test, model.predict(X_test)))


def rf_objective(trial, X_train, y_train, X_test, y_test):
    n_estimators = trial.suggest_int('n_estimators', 10, 5000)
    max_depth = trial.suggest_int('max_depth', 1, 32)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)

    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                  min_samples_split=min_samples_split, n_jobs=46)

    model.fit(X_train, y_train)

    return (mean_absolute_error(y_test, model.predict(X_test)),
            mean_squared_error(y_test, model.predict(X_test)),
            r2_score(y_test, model.predict(X_test)))


def mlp_objective(trial, X_train, y_train, X_test, y_test):
    n_layers = trial.suggest_int('n_layers', 1, 10)
    layers = []
    for i in range(n_layers):
        layers.append(trial.suggest_int(f'n_units_{i}', 1, 250))

    learning_rate_init = trial.suggest_float('learning_rate_init', 1e-8, 1e-1, log=True)
    max_iter = trial.suggest_int('max_iter', 200, 2000)
    beta_1 = trial.suggest_float('beta_1', 0.5, 0.9)
    beta_2 = trial.suggest_float('beta_2', 0.9, 0.9999)
    batch_size = trial.suggest_int('batch_size', 8, 128, step=8)

    model = MLPRegressor(hidden_layer_sizes=tuple(layers), learning_rate_init=learning_rate_init, max_iter=max_iter,
                          beta_1=beta_1, beta_2=beta_2, batch_size=batch_size, solver='adam', activation='tanh')

    model.fit(X_train, y_train)

    return (mean_absolute_error(y_test, model.predict(X_test)),
            mean_squared_error(y_test, model.predict(X_test)),
            r2_score(y_test, model.predict(X_test)))
