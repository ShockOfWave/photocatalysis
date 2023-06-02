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

def load_les():
    with open(os.path.join(get_project_path(), 'models', 'preprocessing', 'le_dob.pkl'), 'rb') as f:
        le_dob = pickle.load(f)
        f.close()
    
    with open(os.path.join(get_project_path(), 'models', 'preprocessing', 'le_donor.pkl'), 'rb') as f:
        le_donor = pickle.load(f)
        f.close()
        
    with open(os.path.join(get_project_path(), 'models', 'preprocessing', 'le_method.pkl'), 'rb') as f:
        le_method = pickle.load(f)
        f.close()
        
    with open(os.path.join(get_project_path(), 'models', 'preprocessing', 'le_prec.pkl'), 'rb') as f:
        le_prec = pickle.load(f)
        f.close()  
    
    return le_dob, le_donor, le_method, le_prec