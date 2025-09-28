import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

df = pd.read_csv("clean_loan_data.csv")

x = df.drop(columns="loan_status")
y = df['loan_status']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('ohe', OneHotEncoder(), ["education", "self_employed"]),
        ('scaler', MinMaxScaler(), ["no_of_dependents", "income_annum", "loan_amount","loan_term","cibil_score","residential_assets_value","commercial_assets_value","luxury_assets_value","bank_asset_value"])
    ]
)

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeClassifier())
])


pipe.fit(x_train, y_train)

pred = pipe.predict(x_test)

accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred, pos_label='Approved')
recall = recall_score(y_test, pred, pos_label='Approved')
f1 = f1_score(y_test, pred, pos_label='Approved')
cm = confusion_matrix(y_test, pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", cm)

with open("model.pkl", "wb") as f:
    pickle.dump(pipe, f)