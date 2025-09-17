import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the model
model = joblib.load('salary_perdict_model.pkl')

# App Title
st.title("Salary Prediction")
st.markdown("""
Predict salary based on age and eductions.
""")

def user_input_features():
    age = st.slider('Age', 18, 100, 51) 
    education_bachelors = st.radio('Education: Bachelors', [0, 1], index=0)
    education_diploma = st.radio('Education: Diploma', [0, 1], index=0)
    education_doctorate = st.radio('Education: Doctorate', [0, 1], index=0)
    education_masters = st.radio('Education: Masters', [0, 1], index=0)
    education_professional = st.radio('Education: Professional', [0, 1], index=0)

    input_data = pd.DataFrame({
         'Age': [age],
        'Education_Bachelors': [education_bachelors],
        'Education_Diploma': [education_diploma],
        'Education_Doctorate': [education_doctorate],
        'Education_Masters': [education_masters],
        'Education_Professional': [education_professional]
    })

    return input_data


user_input = user_input_features()

st.write("User Input Features: ")
st.write(user_input)

prediction = model.predict(user_input)

st.write(f"Predicted Salary: $ {prediction[0]:,.2f}")