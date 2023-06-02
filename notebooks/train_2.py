import pandas as pd
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import json

df = pd.read_csv('../data/processed/data.csv')

X_data = df.drop(df.columns[-1], axis=1).values
y_data = df[df.columns[-1]].values

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=2)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

with open('../models/preprocessing/sc.pkl', 'wb') as f:
    pickle.dump(sc, f)
    f.close()
    
model = CatBoostRegressor(task_type='GPU', devices='1')

grid = {
    'iterations': [5, 10, 50, 100, 500, 1000, 5000],
    'learning_rate': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    'depth': [1, 3, 5, 7],
    #'l2_leaf_reg': [1, 5, 15, 20, 25],
    'early_stopping_rounds': [500],
    'verbose': [500]
}

grid_search_results = model.grid_search(grid, X_train, y_train, cv=3, train_size=0.8, refit=True)

model.save_model('../models/serialized/model')

json.dump(grid_search_results, open('../data/trained/grid_search_results.json', 'w'), indent=4)

json.dump(grid, open('../data/trained/grid_values.json', 'w'), indent=4)
