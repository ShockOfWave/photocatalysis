import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import PATH_TO_DATA


def split_and_save_data(path: str):
    """
    Split processed data and save it
    Args:
        path (str): path to processed data

    Returns:
        None
    """
    df = pd.read_csv(path)
    x_data = df.drop(df.columns[-1], axis=1).values
    y_data = df[df.columns[-1]].values

    savepath = os.path.join(PATH_TO_DATA, 'processed')

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=2, test_size=0.2)

    np.save(os.path.join(savepath, 'x_train.npy'), x_train, allow_pickle=True)
    np.save(os.path.join(savepath, 'x_test.npy'), x_test, allow_pickle=True)
    np.save(os.path.join(savepath, 'y_train.npy'), y_train, allow_pickle=True)
    np.save(os.path.join(savepath, 'y_test.npy'), y_test, allow_pickle=True)
