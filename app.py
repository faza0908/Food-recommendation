import streamlit as st
import pandas as pd
import tensorflow as tf
from food_recom_model import FoodlensModel
import numpy as np

# Load CSV data
user_data_raw = pd.read_csv("user_data.csv")
food_data_raw = pd.read_csv("food_raw.csv")
food_popularity_raw = pd.read_csv("food_popularity.csv")

# Cek data
food_data = food_data_raw.copy()
food_data['Food_ID'] = food_data['Food_ID'].astype(str)

# Load Model
model = FoodlensModel(layer_sizes=None, popular_weight=1, retrieval_weight=1)
model.load_weights('./saved_model/model_weight').expect_partial()  # Pastikan ada model weight

st.title("🍱 Food Recommendation System for Personalized Nutrition")

# Input User ID
user_id = st.selectbox("Select User ID", user_data_raw["User_ID"].unique())
age = st.number_input("Age", min_value=10, max_value=80, value=int(user_data_raw[user_data_raw["User_ID"]==user_id]["Age"].values[0]))
weight = st.number_input("Body Weight (kg)", min_value=30, max_value=200, value=int(user_data_raw[user_data_raw["User_ID"]==user_id]["Body_Weight"].values[0]))
height = st.number_input("Body Height (cm)", min_value=120, max_value=220, value=int(user_data_raw[user_data_raw["User_ID"]==user_id]["Body_Height"].values[0]))
cal_need = st.number_input("Calorie Need", min_value=1000, max_value=4000, value=int(user_data_raw[user_data_raw["User_ID"]==user_id]["Cal_Need"].values[0]))
gender = st.selectbox("Gender", ["M", "F"], index=["M", "F"].index(user_data_raw[user_data_raw["User_ID"]==user_id]["sex"].values[0]))

# Predict Button
if st.button("Recommend Foods"):
    input_dict = {
        "User_ID": tf.constant([user_id]),
        "Age": tf.constant([age]),
        "Body_Weight": tf.constant([weight]),
        "Body_Height": tf.constant([height]),
        "Cal_Need": tf.constant([cal_need]),
        "sex": tf.constant([gender])
    }

    # Manual Prediction Simulation (Top 10 makanan populer)
    top_foods = food_data_raw.head(10)  # Contoh dummy
    st.write("Top Food Recommendations:")
    st.table(top_foods[["Food_ID", "Nama_Bahan_Makanan"]])

st.info("Note: Model loaded successfully. Pastikan file saved_model/ ada di project.")
