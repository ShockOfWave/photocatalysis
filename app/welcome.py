import streamlit as st
from src.utils import check_password


if not check_password():
    st.stop()


st.set_page_config(
    page_title='Photocatalysis',
    page_icon='🌼',
)

st.write('# Welcome to application')

st.markdown(
    """
    This application was created for testing model for our Data Science project 🔥
    
    **Select any tab in the left side for see our model**

    ### About us
    - We are from [ITMO University](https://itmo.ru) and [Infochemistry Scientific Center (ISC)](https://infochemistry.ru)
    - And from [Boreskov Institute of Catalysis](https://catalysis.ru)
    
    ### Introduction
    In this study, a machine learning method was applied to determine the optimal conditions for the synthesis of graphitic carbon nitride (g-C3N4) for use in photocatalytic hydrogen production processes. The main concept of this work is that predictive machine learning algorithms make it possible to develop a method for assessing the photocatalytic activity of a catalyst, which in turn reduces the cost of resources and effort compared to traditional computational and/or experimental methods. In connection with this task, a database was experimentally formed by obtaining g-C3N4 samples by heat treatment of nitrogen-containing precursors in air at a temperature of 450–600°C with varying synthesis heating times and rates.. Graphitic carbon nitride based materials were characterized by physicochemical analysis methods including X-ray phase analysis and low temperature nitrogen adsorption.Based on the obtained experimental data, a table for machine learning was generated. The results of the study showed that the use of machine learning can significantly improve the synthesis process of g-C3N4 material and increase its efficiency in the process of photocatalytic hydrogen evolution. This means that through the use of machine learning, more accurate predictions of the properties of graphitic carbon nitride depending on the parameters of its synthesis, which in turn can lead to the directed synthesis of a highly active material based on g-C3N4 used as catalysts in the reaction of photocatalytic hydrogen production.
    """
)