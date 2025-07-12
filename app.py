# churn_prediction_system.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from xgboost import XGBClassifier
import joblib



import pandas as pd


df = pd.read_csv('easy_churn_data.csv') 


df.columns = df.columns.str.strip()


if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

print(" Dataset loaded and cleaned (without 'TotalCharges').")
print(df.head())



le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])


X = df.drop("Churn", axis=1)
y = df["Churn"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)


joblib.dump(model, "churn_model.pkl")
print("Model Trained and Saved as churn_model.pkl")

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred,
    'Churn_Probability': y_proba
})
results_df.to_csv("churn_predictions.csv", index=False)
print("Predictions saved to churn_predictions.csv")

plt.figure(figsize=(10, 6))
feat_imp = model.feature_importances_
features = X.columns
sns.barplot(x=feat_imp, y=features)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()


import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [10, 20, 25, 30, 40]


plt.plot(x, y, marker='o')
plt.title("Sample Line Chart")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")


plt.savefig("my_plot.png")  
print("Plot saved as 'my_plot.png'")
