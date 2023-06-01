import os
import pickle
from catboost import CatBoostRegressor
from src.utils import get_project_path

def load_model():
    model = CatBoostRegressor(task_type='GPU', devices='1')
    model.load_model(os.path.join(get_project_path(), 'models', 'serialized', 'model'))
    return model

def predict_model(model, data):
    return model.predict(data)

def load_sc():
    with open(os.path.join(get_project_path(), 'models', 'preprocessing', 'sc.pkl'), 'rb') as f:
        sc = pickle.load(f)
        f.close()
    
    return sc