import os
import pandas as pd
import streamlit as st
from src.utils import PATH_TO_REPORTS
from src.utils import get_project_path
from src.models.predict_model import load_model, load_process
from src.utils import check_password


if not check_password():
    st.stop()


@st.cache_data
def read_metrics():
    df = pd.read_csv(os.path.join(PATH_TO_REPORTS, 'tables', 'final_df.csv'))
    return df


def read_data():
    df = pd.read_csv(os.path.join(get_project_path(), 'data', 'processed', 'data.csv'))
    return df


df = read_data()
sc = load_process()
model = load_model()
metrics = read_metrics()


st.markdown('# Train')


st.markdown(
    """
    ### On this page we show you evaluation of trained model
    We used 3 algorithms: **Multilayer perceptron(MLP)**, Random Forest(RF) and Gradient boosting
    
    MLP and RF created by [Scikit-Learn](https://scikit-learn.org/stable/) library, gradient booting by [CatBoost](https://catboost.ai)
    
    Hyperparameter optimization was made by [Optuna](https://optuna.org)
    
    On this page you can see table with metrics of all model with different type of processing and some plots with feature importances and model evaluation
    
    ### Table with metrics
    """
)

st.table(metrics)

st.markdown(
    f"""Table shows that best results were made by Gradient Boosting and MLP. 
    We deside use Gradient boosting, 
    because is has less error on train dataset and 
    better worked with data without process."""
)

y_true = df[df.columns[-1]].values

pred = model.predict(sc.transform(df.drop(df.columns[-1], axis=1)))

st.markdown(f"""
            ### Best model metrics
            Best model shows us very good metrics:
            
            $MAE = 0.1382$
            
            $MSE = 0.0473$
            
            $R^2 = 0.9901$
            
            ##### Regression plot
            On the regression plot you can see dependency between True values and Predicted values.
            """)
st.image(
    os.path.join(PATH_TO_REPORTS, 'figures', 'catboost', 'norm_data', 'test_reg_plot.svg'),
    caption='Regression plot'
)

st.markdown(f"""
### Feature importance
For model analysis used [SHAP](https://github.com/shap/shap) library

Feature importance shows us that the almost all features are using for model predictions, 
most important **XRD angles**, **catalytic activity** and **synthesis parameters**
""")

st.image(
    os.path.join(PATH_TO_REPORTS, 'figures', 'catboost', 'norm_data', 'beeswarm.svg'),
    caption='Shap beeswarm plot'
)
