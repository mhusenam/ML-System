import mlflow
import pandas as pd
import joblib
import numpy as np
import pandas as pd
import requests
import json

def prediction(data):
    """Making prediction
 
    Args:
        data (Pandas DataFrame): Dataframe that contain all the preprocessed data
 
    Returns:
        str: Prediction result (Good, Standard, or Poor)
    """
    # URL endpoint dari model yang sedang di-serve
    url = "http://127.0.0.1:5001/invocations"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, data=data, headers=headers)
    response = response.json().get("predictions")
    result_target = joblib.load("model/encoder_target.joblib")
    final_result = result_target.inverse_transform(response)
    return final_result

columns = [
    "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour", "Age", "Num_Bank_Accounts", "Num_Credit_Card",
    "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
    "Num_Credit_Inquiries", "Outstanding_Debt", "Monthly_Inhand_Salary", "Monthly_Balance",
    "Amount_invested_monthly", "Total_EMI_per_month", "Credit_History_Age"
]

data = ["Good","No","Low_spent_Small_value_payments",23,3,4,3,4,3,7,11.27,5,809.98,1824.80,186.26,236.64,49.50,216]

df = pd.DataFrame([data], columns=columns)
new_data = data_preprocessing(data=df)

# Konversi DataFrame ke format JSON yang diinginkan
json_output = {
    "dataframe_split": {
        "columns": new_data.columns.tolist(),
        "data": new_data.values.tolist()
    }
}
data_testing = json.dumps(json_output)
print(prediction(data_testing))