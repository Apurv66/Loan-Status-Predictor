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

---
## Screenshots
<img width="400" height="250" alt="Screenshot 2025-09-28 111225" src="https://github.com/user-attachments/assets/e77b4f52-0e88-4dd2-aeb7-6c8e5790458f" />
<img width="400" height="250" alt="Screenshot 2025-09-28 111333" src="https://github.com/user-attachments/assets/7028c2de-26ec-4ef4-9012-158b406ba1c7" />
