import pandas as pd
import streamlit as st
import pickle
import numpy as np

pipe = pickle.load(open("model.pkl", "rb"))


df = pd.read_csv("clean_loan_data.csv")

st.title("Loan Approval Prediction")
st.divider()

dependents = st.number_input("Enter number of dependents:",min_value=0 ,step=1)
education = st.selectbox("Enter education:", df['education'].unique())
isEmployeed = st.selectbox("Are you self-employed?", df['self_employed'].unique())
annual_income = st.number_input("Enter annual income:",min_value=0 ,step=5000)
loan_amount = st.number_input("Enter loan amount:",min_value=0 ,step=5000)
term = st.number_input("Enter loan term (in year):",min_value=0 ,step=1)
cibil = st.number_input("Enter CIBIL score:",min_value=0 ,step=10)
residential_asset = st.number_input("Enter residential assets value:",min_value=0 ,step=5000)
commercial_asset = st.number_input("Enter commercial assets value:",min_value=0 ,step=5000)
luxury_asset = st.number_input("Enter luxury assets value:",min_value=0 ,step=5000)
bank_asset = st.number_input("Enter bank asset value:",min_value=0 ,step=5000)

input_data = pd.DataFrame([[
    dependents, education, isEmployeed, annual_income, loan_amount,
    term, cibil, residential_asset, commercial_asset, luxury_asset, bank_asset
]], columns=df.columns.drop("loan_status"))

if cibil < 300:
    st.warning("CIBIL score below 300 is invalid.")
else:
    if st.button("Predict"):
        try:
            pred = pipe.predict(input_data)
            if pred[0]=="Approved":
                st.success("Loan Status: Approved")
            else:
                st.warning("Loan Status: Rejected")
        except Exception as e:
            st.error(f"Error in prediction: {e}")