import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle

# Load data
user_data_raw = pd.read_pickle("./user_data.pkl")
food_data_raw = pd.read_pickle("./food_raw.pkl")
food_popularity_raw = pd.read_pickle("./food_popularity.pkl")

# Load model
from Demo_deploy_with_gradio import FoodlensModel, predict_food, top_nutrition

model = FoodlensModel(layer_sizes=None,popular_weight=1, retrieval_weight=1)
model.load_weights('./saved_model/model_weight').expect_partial()

st.title("🍱 Sistem Rekomendasi Makanan Berdasarkan Profil Pribadi")

# Input User
User_ID = st.text_input("User ID", "UNT001")
Age = st.number_input("Age", 18, 80, 25)
Body_Weight = st.number_input("Body Weight (kg)", 40, 150, 65)
Body_Height = st.number_input("Body Height (cm)", 140, 200, 170)
Cal_Need = st.number_input("Calories Need", 1200, 4000, 2000)
Gender = st.selectbox("Gender", ["M", "F"])
Amount_Of_Eat = st.selectbox("Amount of Eat (2/3/4 meals)", [2,3,4], index=1)

if st.button("Generate Food Recommendations"):
    # Run prediction
    input_dict = {
        "User_ID": tf.constant([User_ID]),
        "Age": tf.constant([Age]),
        "Body_Weight": tf.constant([Body_Weight]),
        "Body_Height": tf.constant([Body_Height]),
        "Cal_Need": tf.constant([Cal_Need]),
        "sex": tf.constant([Gender])
    }

    food_recom = predict_food(
        food_data_raw=food_data_raw,
        input_dict=input_dict,
        output_type="dict",
        model_recom=model,
        top_n=15
    )
    food_df = pd.DataFrame(food_recom)
    food_df.columns = ["Index", "Food ID", "Food Name"]
    st.subheader("Rekomendasi Makanan (Top 15)")
    st.dataframe(food_df)

    top_nutri = top_nutrition(
        food_data_raw=food_data_raw,
        user_id=User_ID,
        list_recom_food=list(food_df["Food Name"]),
        gender=Gender,
        pred_cal=Cal_Need,
        amount_of_eat=Amount_Of_Eat
    )
    nutri_df = pd.DataFrame(top_nutri).T
    st.subheader("Top Nutrisi Kombinasi (Rekomendasi Harian)")
    st.dataframe(nutri_df)

