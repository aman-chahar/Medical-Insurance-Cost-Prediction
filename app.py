import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the trained model
pickle_in = open("regressor.pkl", "rb")
regressor = pickle.load(pickle_in)

# App title
st.title("Insurance Cost Prediction")

# Sidebar with user input
st.sidebar.header("User Input Features")

# Age
age = st.sidebar.slider("Age", 18, 64, 30)

# Sex
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])

# BMI
bmi = st.sidebar.slider("BMI (Body Mass Index)", 10.0, 50.0, 25.0)

# Children
children = st.sidebar.slider("Number of Children/Dependents", 0, 10, 0)

# Smoker
smoker = st.sidebar.selectbox("Smoker", ["No", "Yes"])

# Region
region = st.sidebar.selectbox("Region", ["Southeast", "Southwest", "Northeast", "Northwest"])

# Encoding categorical inputs
sex = 1 if sex == "Female" else 0
smoker = 0 if smoker == "Yes" else 1
region_encoded = {"Southeast": 0, "Southwest": 1, "Northeast": 2, "Northwest": 3}
region = region_encoded[region]

# Create feature vector
features = np.array([age, sex, bmi, children, smoker, region]).reshape(1, -1)

# Make prediction
if st.button("Predict"):
    prediction = regressor.predict(features)
    st.header("Prediction")
    st.write(f"The estimated insurance cost is USD {prediction[0]:.2f}")

# Model performance metrics
st.sidebar.title("Model Performance Metrics")


# Input data display
st.sidebar.title("Input Data")
st.sidebar.write("Input data used for the prediction:")
st.sidebar.write(f"Age: {age}")
st.sidebar.write(f"Sex: {sex}")
st.sidebar.write(f"BMI: {bmi}")
st.sidebar.write(f"Children: {children}")
st.sidebar.write(f"Smoker: {smoker}")
st.sidebar.write(f"Region: {region}")

# Display dataset analysis and visualizations (optional)
st.sidebar.title("Dataset Analysis and Visualizations")
st.sidebar.write("Dataset analysis and visualizations can be displayed here.")

