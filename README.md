# Loan Predictor App

A **Supervised Machine Learning Classification Project** that predicts **Loan Status (Approved/Rejected)** based on applicant details like dependents, income, loan amount, credit score, assets, education, and employment status.  

This project demonstrates the **end-to-end ML pipeline**:  
`data cleaning → preprocessing → model training → evaluation → deployment with Streamlit`.  

---

## Data Preparation  

- Removed extra spaces from column names and string values  
- Dropped unnecessary `loan_id` column  
- Replaced negative values in `residential_assets_value`  
- Encoded categorical features (`education`, `self_employed`) with **LabelEncoder**  
- Scaled numerical features with **MinMaxScaler**  

---

## Model Training  

- Algorithm: **DecisionTreeClassifier**  
- Dataset split using **train_test_split**  
- Evaluation metrics:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  
  - Confusion Matrix  

## Tech Stack  

- **Python**  
- **pandas**, **numpy** → data processing  
- **scikit-learn** → preprocessing, model training, evaluation  
- **Streamlit** → web app interface  
- **pickle** → saving models & encoders  

