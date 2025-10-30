import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
import json

# --- 1. SET PAGE CONFIG (Sets White Theme & Icon) ---
# This must be the first Streamlit command.
st.set_page_config(
    page_title="Credit Card Approval",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. LOAD YOUR TRAINED MODEL AND PREPROCESSOR ---
# We use @st.cache_data to load this only once.
@st.cache_data
def load_model_and_preprocessor():
    try:
        preprocessor = pickle.load(open("preprocessor.pkl", "rb"))
        model = pickle.load(open("best_model.pkl", "rb"))
        return preprocessor, model
    except FileNotFoundError:
        st.error("Error: `preprocessor.pkl` or `best_model.pkl` not found.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading files: {e}")
        st.stop()

preprocessor, model = load_model_and_preprocessor()

# Get the list of features the model was trained on
# This is a robust way to ensure all columns are present
try:
    cat_features = preprocessor.named_transformers_['cat'].feature_names_in_
    num_features = preprocessor.named_transformers_['num'].feature_names_in_
    model_features = list(num_features) + list(cat_features)
except Exception:
    # Fallback if the above fails (e.g., older scikit-learn version)
    st.warning("Could not automatically get feature names. Using a hardcoded list.")
    # You may need to manually update this list to match your notebook
    model_features = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 
                      'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL', 
                      'CNT_FAM_MEMBERS', 'month_from_today', 'CODE_GENDER', 
                      'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 
                      'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE']


# --- 3. SIDEBAR FOR USER INPUTS ---
# This keeps your main page clean and professional.
st.sidebar.header("Applicant Information")

with st.sidebar:
    # --- Numerical Inputs ---
    # We need to add all inputs your model was trained on.
    # I've kept the ones you had and added the missing ones.
    
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    # We convert age to DAYS_BIRTH for the model
    days_birth = age * -365 
    
    income = st.number_input("Total Annual Income", min_value=10000, value=150000, step=1000)
    
    days_employed_input = st.number_input("Days Employed (Enter 0 if unemployed)", min_value=0, max_value=20000, value=365)
    # Convert to negative as per your dataset
    days_employed = days_employed_input * -1 if days_employed_input > 0 else 365243 # 365243 is the "unemployed" flag in the notebook
    
    children = st.number_input("Number of Children", min_value=0, max_value=20, value=0)
    fam_members = st.number_input("Number of Family Members", min_value=1, max_value=21, value=children + 1)

    # --- Categorical Inputs (using selectbox) ---
    gender = st.selectbox("Gender", ["M", "F"])
    car_owner = st.selectbox("Owns a Car?", ["Y", "N"])
    realty_owner = st.selectbox("Owns Real Estate?", ["Y", "N"])
    
    income_type = st.selectbox("Income Type", [
        'Working', 'Commercial associate', 'Pensioner', 'State servant', 'Student'
    ])
    
    education_type = st.selectbox("Education Level", [
        'Higher education', 'Secondary / secondary special', 'Incomplete higher', 
        'Lower secondary', 'Academic degree'
    ])
    
    family_status = st.selectbox("Marital Status", [
        'Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'
    ])
    
    housing_type = st.selectbox("Housing Type", [
        'House / apartment', 'With parents', 'Municipal apartment', 
        'Rented apartment', 'Office apartment', 'Co-op apartment'
    ])
    
    # --- Hidden inputs (defaults your model needs) ---
    flag_mobil = 1
    flag_work_phone = 0
    flag_phone = 0
    flag_email = 0
    month_from_today = 0 # Placeholder for the 'month_from_today' feature


# --- 4. MAIN PAGE FOR DISPLAYING RESULTS ---
st.title("💳 Credit Card Approval Prediction")
st.write("This app uses a machine learning model to predict whether a credit card application will be approved or rejected.")

# Layout for the result
col1, col2 = st.columns([1, 1])

# The "Predict" button is now in the sidebar
if st.sidebar.button("Predict Approval", type="primary"):
    
    # Create a DataFrame from the inputs
    # The column names MUST match the ones your model was trained on
    input_data = pd.DataFrame({
        'CODE_GENDER': [gender],
        'FLAG_OWN_CAR': [car_owner],
        'FLAG_OWN_REALTY': [realty_owner],
        'CNT_CHILDREN': [children],
        'AMT_INCOME_TOTAL': [income],
        'NAME_INCOME_TYPE': [income_type],
        'NAME_EDUCATION_TYPE': [education_type],
        'NAME_FAMILY_STATUS': [family_status],
        'NAME_HOUSING_TYPE': [housing_type],
        'DAYS_BIRTH': [days_birth],
        'DAYS_EMPLOYED': [days_employed],
        'FLAG_MOBIL': [flag_mobil],
        'FLAG_WORK_PHONE': [flag_work_phone],
        'FLAG_PHONE': [flag_phone],
        'FLAG_EMAIL': [flag_email],
        'CNT_FAM_MEMBERS': [fam_members],
        'month_from_today': [month_from_today] 
    }, columns=model_features) # Ensure correct column order

    with col1:
        st.header("Prediction Result")
        with st.spinner("Analyzing applicant..."):
            
            try:
                # --- 5. PREPROCESS AND PREDICT ---
                
                # Apply the preprocessor
                input_processed = preprocessor.transform(input_data)
                
                # Make prediction (your model outputs 1 for 'Approved', 0 for 'Rejected')
                prediction = model.predict(input_processed)
                
                # Get prediction probability
                probability = model.predict_proba(input_processed)
                
                # --- 6. DISPLAY RESULTS ---
                if prediction[0] == 1:
                    st.success("🎉 **Approved**")
                    st.write(f"Confidence: **{probability[0][1]*100:.2f}%**")
                    st.progress(probability[0][1])
                    st.balloons()
                    
                else:
                    st.error("❌ **Rejected**")
                    st.write(f"Confidence: **{probability[0][0]*100:.2f}%**")
                    st.progress(probability[0][0])

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

with col2:
    st.image("https://images.unsplash.com/photo-1553729459-FAB0a5a528a5?ixlib=rb-4.0.3&q=80&fm=jpg&crop=entropy&cs=tinysrgb&w=1080&fit=max",
             caption="Credit card approval prediction",
             use_column_width=True)

# --- 7. A NOTE ON YOUR API KEY ---
st.sidebar.title("") # Add some space
st.sidebar.info(
    "**Note on API Keys:** Your original app had an API key visible in the code. "
    "**Never save API keys directly in your script!** "
    "Use Streamlit's Secrets Management to keep them safe. "
    "I have removed the Gemini API call and am using your trained ML model instead."
)