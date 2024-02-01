import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils import PATH_TO_MODELS, PATH_TO_DATA


def read_and_process_data(path: str) -> pd.DataFrame:
    """
    Read raw data, process it, saves label encoders and data
    Args:
        path (str): Path to raw data

    Returns:
        dataframe (pd.DataFrame): returns processed dataset
    """
    df = pd.read_csv(path, encoding='latin', delimiter=';')
    df.drop(df.columns[0], axis=1, inplace=True)

    save_path = os.path.join(PATH_TO_MODELS, 'process')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    le_prec = LabelEncoder()
    le_dob = LabelEncoder()
    le_method = LabelEncoder()
    le_donor = LabelEncoder()

    df['Precursor'] = le_prec.fit_transform(df['Precursor'])
    df['Additive'] = le_dob.fit_transform(df['Additive'])
    df['Application method of Pt'] = le_method.fit_transform(df['Application method of Pt'])
    df['Electron donor'] = le_donor.fit_transform(df['Electron donor'])

    pickle.dump(le_prec, open(os.path.join(save_path, 'le_prec.pkl'), 'wb'))
    pickle.dump(le_dob, open(os.path.join(save_path, 'le_dob.pkl'), 'wb'))
    pickle.dump(le_method, open(os.path.join(save_path, 'le_method.pkl'), 'wb'))
    pickle.dump(le_donor, open(os.path.join(save_path, 'le_donor.pkl'), 'wb'))

    if not os.path.exists(os.path.join(PATH_TO_DATA, 'processed')):
        os.makedirs(os.path.join(PATH_TO_DATA, 'processed'))

    df.to_csv(os.path.join(PATH_TO_DATA, 'processed', 'data.csv'), index=False)

    return df
