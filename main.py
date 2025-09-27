import pandas as pd
import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
selfemp_encoder = pickle.load(open("self_employed_encoder.pkl", "rb"))
edu_encoder = pickle.load(open("education_encoder.pkl", "rb"))

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

education_encoded = edu_encoder.transform([education])[0]
self_employed_encoded = selfemp_encoder.transform([isEmployeed])[0]

numeric_inputs = np.array([[dependents, annual_income, loan_amount, term, cibil, residential_asset, commercial_asset, luxury_asset, bank_asset]])
scaled_inputs = scaler.transform(numeric_inputs)

final_input = np.hstack((scaled_inputs, [[education_encoded, self_employed_encoded]]))

if cibil < 300:
    st.warning("CIBIL score below 300 is invalid.")
else:
    if st.button("Predict"):
        try:
            pred = model.predict(final_input)
            if pred[0]=="Approved":
                st.success("Loan Status: Approved")
            else:
                st.warning("Loan Status: Rejected")
        except Exception as e:
            st.error(f"Error in prediction: {e}")