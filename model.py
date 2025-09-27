import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("clean_loan_data.csv")

catogorical_cols = ['education', 'self_employed']

encoders = {}

for i in catogorical_cols:
    le = LabelEncoder()
    df[i] = le.fit_transform(df[i])
    encoders[i] = le

cols = df.columns
cols = cols.drop(['education', 'self_employed', 'loan_status'])

scaler = MinMaxScaler()
df[cols] = scaler.fit_transform(df[cols])

x = df.drop(columns="loan_status").values
y = df['loan_status'].values

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()

model.fit(x_train, y_train)

pred = model.predict(x_test)

accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred, pos_label='Approved')
recall = recall_score(y_test, pred, pos_label='Approved')
f1 = f1_score(y_test, pred, pos_label='Approved')
cm = confusion_matrix(y_test, pred)
print(accuracy)
print(precision)
print(recall)
print(f1)
print(cm)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl","wb") as f:
    pickle.dump(scaler,f)

for col, le in encoders.items():
    pickle.dump(le, open(f"{col}_encoder.pkl", "wb"))