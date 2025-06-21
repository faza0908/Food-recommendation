%%writefile app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="Online Payment Fraud Detection", layout="wide")

st.title("Online Payment Fraud Detection")

# Load the saved model and preprocessors
loaded_model = joblib.load('rfc_model.joblib')

# Recreate LabelEncoder and StandardScaler with the same fitting as before
# Since the original training data is not directly available here,
# we'll simulate fitting on the types and a sample of numerical data structure
# (In a real application, you would save and load the fitted preprocessors)

# For LabelEncoder, we know the categories are 'PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'
le = LabelEncoder()
le.fit(['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'])

# For StandardScaler, we need the mean and std dev from the training data.
# As a workaround for this example, we'll use dummy data structure.
# In a real scenario, save and load the fitted scaler: joblib.dump(scaler, 'scaler.joblib')
# and joblib.load('scaler.joblib')
# Let's assume the scaler was fitted on columns:
# 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'
# We need the mean and std_dev for each of these columns from the *training* data.
# Since we don't have the original scaler or training data stats saved, this is a placeholder.
# A real solution requires saving and loading the fitted scaler during model training.
# For demonstration, we will create a dummy scaler.
# THIS IS A SIMPLIFICATION FOR DEMONSTRATION ONLY.
# A proper implementation requires saving and loading the fitted scaler.

# Dummy data for scaler fitting structure - Replace with actual loaded scaler
dummy_data = pd.DataFrame({
    'type': [0],
    'amount': [0.0],
    'oldbalanceOrg': [0.0],
    'newbalanceOrig': [0.0],
    'oldbalanceDest': [0.0],
    'newbalanceDest': [0.0]
})
scaler = StandardScaler()
scaler.fit(dummy_data) # Fit with dummy data structure


st.header("Enter Transaction Details")

# Input for 'type' (categorical)
transaction_type = st.selectbox(
    "Transaction Type",
    ('PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN')
)

# Input for numerical features
amount = st.number_input("Amount", min_value=0.0, format="%.2f")
oldbalanceOrg = st.number_input("Old Balance Originator", min_value=0.0, format="%.2f")
newbalanceOrig = st.number_input("New Balance Originator", min_value=0.0, format="%.2f")
oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0, format="%.2f")
newbalanceDest = st.number_input("New Balance Destination", min_value=0.0, format="%.2f")

# Prediction button
if st.button("Predict Fraud"):
    # Collect user input into a DataFrame
    user_input = pd.DataFrame({
        'type': [transaction_type],
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest]
    })

    # Apply preprocessing
    # Label encode the 'type' column
    user_input['type'] = le.transform(user_input['type'])

    # Scale the numerical features
    # IMPORTANT: In a real application, load the fitted scaler used during training.
    # The dummy scaler here will not produce correct results.
    scaled_input = scaler.transform(user_input)


    # Make prediction
    prediction = loaded_model.predict(scaled_input)

    # Display result
    if prediction[0] == 1:
        st.error("Fraudulent Transaction Detected!")
    else:
        st.success("Transaction is Not Fraudulent.")
