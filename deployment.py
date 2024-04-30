import streamlit as st
import numpy as np
import pandas as pd
from predict import load_from_pkl, scale, model_predict


def make_prediction(df):
    scaler = load_from_pkl('scaler.pkl')
    model = load_from_pkl('xgb.pkl')

    df.loc[:, ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]] = scale(df.loc[:, ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]], scaler)
    
    return model_predict(df, model)

def main():
    st.title("Churn Prediction")

    creditscore = st.number_input("Credit Score", step=1)
    gender = st.radio("Gender", ["Male", "Female"])
    age = st.number_input("Age", step=1)
    tenure = st.number_input("Tenure", step=1)
    balance = st.number_input("Balance")
    numofproducts = st.number_input("Number of Products", step=1)
    hascrcard = st.radio("Do you have a credit card : ", ["Yes", "No"])
    isactivemember = st.radio("Are you an active member : ", ["Yes", "No"])
    estimatedsalary = st.number_input("EstimatedSalary")
    geography = st.radio("Geography : ", ["France", "Germany", "Spain"])


    data = {"CreditScore":[int(creditscore)], 'Gender': [0 if gender=="Male" else 1], 'Age':[int(age)], 'Tenure':[int(tenure)], 'Balance':[float(balance)],
            'NumOfProducts': [int(numofproducts)], 'HasCrCard':[0 if hascrcard == "No" else 1], 'IsActiveMember': [0 if isactivemember == "No" else 1],
            'EstimatedSalary':[float(estimatedsalary)], 'France':[1 if geography=="France" else 0], 'Germany':[1 if geography=="Germany" else 0], 'Spain' : [1 if geography=="Spain" else 0]}
    
    df = pd.DataFrame.from_dict(data)

    if st.button('Predict'):
        res = make_prediction(df)[0]
        st.success(f'Churn : {"Yes" if res==1 else "No"}')


if __name__ == '__main__':
    main()
