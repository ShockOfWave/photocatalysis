import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import time
import os
from src.utils import get_project_path
from sklearn.metrics import *
from src.models.predict_model import load_les, load_model, load_sc
import pickle

plt.style.use('ggplot')

st.set_page_config(
    page_title='Train',
    page_icon="👟",
)

st.markdown('# Train')
st.sidebar.header('Train')

st.write(
    """
    On this page we show you evaluation of trained model
    """
)

st.markdown(
    """
    ### RMSE
    #### RMSE train
    """
)

def read_grid_search_results():
    with open(os.path.join(get_project_path(), 'data', 'trained', 'grid_search_results.json'), 'r') as f:
        grid_search_results = json.loads(f.read())
        f.close()
        
    grid_search_results = pd.json_normalize(grid_search_results)
    
    iterations = grid_search_results['cv_results.iterations'].values[0]
    RMSE_train = grid_search_results['cv_results.train-RMSE-mean'].values[0]
    RMSE_test = grid_search_results['cv_results.test-RMSE-mean'].values[0]
    
    return iterations, RMSE_train, RMSE_test

def plot_RMSE_graph(iterations, RMSE, key=1):
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    chart = st.line_chart([RMSE[0]])
    
    time_of_plotting = 3
    
    time_sleep = time_of_plotting / len(iterations)
    
    progress_delta = 100 // len(iterations)
    
    progress_status = progress_delta
    
    for i in range(len(iterations)-1):
        new_row = [RMSE[i+1]]
        status_text.text("%i%% Complete" % (progress_status))
        chart.add_rows(new_row)
        progress_bar.progress(progress_status)
        time.sleep(time_sleep)
        progress_status+=progress_delta
    
    progress_bar.empty()
    st.button("Re-run", key=key)

iterations, RMSE_train, RMSE_test = read_grid_search_results()
plot_RMSE_graph(iterations, RMSE_train, key=1)

st.markdown(
    """
    #### RMSE test
    """
)

plot_RMSE_graph(iterations, RMSE_test, key=2)

df = pd.read_csv(os.path.join(get_project_path(), 'data', 'processed', 'data.csv'))
sc = load_sc()

y_true = df[df.columns[-1]].values

model = load_model()

pred = model.predict(sc.transform(df.drop(df.columns[-1], axis=1)))

st.markdown(f"""
            ### Metrics
            #### R2 = {r2_score(y_true, pred)}
            #### MAE = {mean_absolute_error(y_true, pred)}
            #### MSE = {mean_squared_error(y_true, pred)}
            """)

fig, ax = plt.subplots(figsize=(15, 9))
ax.scatter(pred, y_true, color='blue')
ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
ax.set_title('Regression model')
st.pyplot(fig)
    