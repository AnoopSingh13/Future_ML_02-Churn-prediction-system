import pandas as pd

data = {
    'CustomerID': ['C001', 'C002', 'C003', 'C004', 'C005', 'C006'],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male'],
    'SeniorCitizen': [0, 1, 0, 1, 0, 0],
    'Tenure': [5, 20, 15, 10, 3, 24],
    'MonthlyCharges': [70.35, 89.10, 65.55, 99.00, 50.25, 85.75],
    'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'Two year', 'Month-to-month', 'One year'],
    'Churn': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data)
df.to_csv("easy_churn_data.csv", index=False)
print("easy_churn_data.csv created successfully")
