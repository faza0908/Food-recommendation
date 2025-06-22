import streamlit as st
import pandas as pd
import tensorflow as tf
from food_recom_model import FoodlensModel
import numpy as np

# Load dataset
user_data = pd.read_csv("user_data.csv")
food_data = pd.read_csv("food_raw.csv")
food_popularity = pd.read_csv("food_popularity.csv")

st.title("🍱 Food Recommendation System")

# Pilih User
user_id = st.selectbox("Select User ID", user_data["User_ID"].unique())

# Ambil data user otomatis
user_info = user_data[user_data["User_ID"] == user_id].iloc[0]
age = user_info["Age"]
weight = user_info["Body_Weight"]
height = user_info["Body_Height"]
cal_need = user_info["Cal_Need"]
sex = user_info["sex"]

# Load Model
model = FoodlensModel(layer_sizes=None, popular_weight=1, retrieval_weight=1)
try:
    model.load_weights('./saved_model/model_weight').expect_partial()
except:
    st.warning("❗ Model weight not found. Using dummy prediction.")

if st.button("Get Food Recommendation"):
    # Contoh dummy output
    rec_food = food_data.head(5)[["Food_ID", "Nama_Bahan_Makanan"]]
    st.write("Recommended Foods:")
    st.dataframe(rec_food)

st.info("This is a demo Streamlit app. Actual model requires trained weights.")
