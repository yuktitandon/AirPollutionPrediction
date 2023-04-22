import streamlit as st
import pandas as pd
import numpy as np
import joblib


model = joblib.load('model.joblib')


def predict_air_pollution(feature1, feature2, feature3, feature4, feature5):
    features = np.array([[feature1, feature2, feature3, feature4, feature5]])
    prediction = model.predict(features)
    return prediction[0]


st.title("Air Pollution Prediction")


feature1 = st.slider('CO', 0.0, 100.0, 25.0)
feature2 = st.slider('NO2', 0.0, 100.0, 25.0)
feature3 = st.slider('O3', 0.0, 100.0, 25.0)
feature4 = st.slider('SO2', 0.0, 100.0, 25.0)
feature5 = st.slider('PM2.5', 0.0, 100.0, 25.0)


if st.button("Predict"):
    result = predict_air_pollution(feature1, feature2, feature3, feature4, feature5)
    st.write("The predicted air pollution level is:", result)

