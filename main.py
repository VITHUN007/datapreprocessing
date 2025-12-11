import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

try:
    with open("titanic_model.pkl", "rb") as file:
        loaded_model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'titanic_model.pkl' not found.")
    st.stop()

try:
    df_cleaned = pd.read_csv("cleaned_titanic_data.csv")
except FileNotFoundError:
    st.error("cleaned_titanic_data.csv not found.")
    st.stop()

numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']

scaler = StandardScaler()
scaler.fit(df_cleaned[numerical_cols])

sex_encoder = LabelEncoder()
sex_encoder.fit(['female', 'male'])  


st.title(" Titanic Survival Prediction App")
st.write("Predict your chance of surviving the Titanic disaster using a Logistic Regression model.")

st.header("Passenger Information")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex_input = st.radio("Sex", ['Male', 'Female'])
age = st.slider("Age", 0.42, 80.0, 30.0)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.number_input("Parents/Children Aboard", 0, 6, 0)
fare = st.number_input("Ticket Fare ($)", 0.0, 512.3292, 32.0)
embarked = st.selectbox("Port of Embarkation", ['Southampton (S)', 'Cherbourg (C)', 'Queenstown (Q)'])


if st.button("Predict Survival"):
    embarked_Q = 1 if embarked == 'Queenstown (Q)' else 0
    embarked_S = 1 if embarked == 'Southampton (S)' else 0 

    pclass_2 = 1 if pclass == 2 else 0
    pclass_3 = 1 if pclass == 3 else 0

    sex_encoded = sex_encoder.transform([sex_input.lower()])[0]

    input_data = pd.DataFrame([[
        sex_encoded, age, sibsp, parch, fare,
        embarked_Q, embarked_S,
        pclass_2, pclass_3
    ]], columns=[
        'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
        'Embarked_Q', 'Embarked_S',
        'Pclass_2', 'Pclass_3'
    ])


    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

    prediction = loaded_model.predict(input_data)[0]
    prediction_proba = loaded_model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success("🎉 You would have SURVIVED!")
        st.write(f"Survival Probability: **{prediction_proba * 100:.2f}%**")
        st.balloons()
    else:
        st.error(" You would NOT have survived.")
        st.write(f"Survival Probability: **{prediction_proba * 100:.2f}%**")
