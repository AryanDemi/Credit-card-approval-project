import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# --- 1. SET PAGE CONFIG (White Background, Wide Layout) ---
st.set_page_config(
    page_title="Credit Card Approval",
    page_icon="💳",
    layout="wide"
)

# --- 2. LOAD YOUR TRAINED MODEL AND PREPROCESSOR ---
@st.cache_data
def load_model_and_preprocessor():
    try:
        # Load the preprocessor
        with open("preprocessor.pkl", "rb") as f_pre:
            preprocessor = pickle.load(f_pre)
        
        # Load the model
        with open("best_model.pkl", "rb") as f_mod:
            model = pickle.load(f_mod)
            
        return preprocessor, model
    except FileNotFoundError:
        st.error("FATAL ERROR: `preprocessor.pkl` or `best_model.pkl` not found.")
        st.info("Please make sure these files are in the same folder as your app.py.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading files: {e}")
        st.stop()

preprocessor, model = load_model_and_preprocessor()

# --- THIS IS THE EXACT FEATURE LIST FROM YOUR FILE ---
NUMERICAL_FEATURES = [
    'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 
    'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL', 
    'CNT_FAM_MEMBERS', 'AGE', 'YEARS_EMPLOYED', 'LOG_INCOME', 'month_from_today'
]
CATEGORICAL_FEATURES = [
    'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 
    'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE'
]
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# --- 3. MAIN PAGE LAYOUT (Inputs in Center) ---
st.title("💳 Credit Card Approval Predictor")
st.markdown("This app uses a trained **K-Nearest Neighbors model** to predict credit card approval.")

col1, col2, col3 = st.columns([1, 1.5, 1])

with col2:
    st.subheader("Applicant Information", divider="blue")
    
    # --- Collect all inputs from the user ---
    r1_col1, r1_col2 = st.columns(2)
    with r1_col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
    with r1_col2:
        income = st.number_input("Total Annual Income", min_value=10000, value=150000, step=1000)
    
    r2_col1, r2_col2 = st.columns(2)
    with r2_col1:
        children = st.number_input("Number of Children", min_value=0, max_value=20, value=0)
    with r2_col2:
        fam_members = st.number_input("Number of Family Members", min_value=1, max_value=21, value=children + 1)

    r3_col1, r3_col2 = st.columns(2)
    with r3_col1:
        car_owner = st.selectbox("Owns a Car?", ["Y", "N"])
    with r3_col2:
        realty_owner = st.selectbox("Owns Real Estate?", ["Y", "N"])

    days_employed_input = st.number_input("Number of Days Employed (Enter 0 if unemployed)", min_value=0, max_value=20000, value=365)
    
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
    
    income_type = st.selectbox("Income Type", [
        'Working', 'Commercial associate', 'Pensioner', 'State servant', 'Student'
    ])
    
    gender = st.selectbox("Gender", ["M", "F"])

    occupation_type = st.selectbox("Occupation Type", [
        'Unknown', 'Laborers', 'Sales staff', 'Core staff', 'Managers', 
        'Drivers', 'High skill tech staff', 'Accountants', 'Medicine staff',
        'Security staff', 'Cooking staff', 'Cleaning staff', 'Private service staff',
        'Low-skill Laborers', 'Waiters/barmen staff', 'Secretaries', 'Realty agents',
        'HR staff', 'IT staff'
    ])

    st.write("") 

    if st.button("Predict Approval", type="primary", use_container_width=True):
        
        # --- 4. PERFORM ALL FEATURE ENGINEERING (This was the missing step) ---
        days_birth = age * -365 
        
        # Handle 'DAYS_EMPLOYED' and 'YEARS_EMPLOYED'
        if days_employed_input > 0:
            days_employed = days_employed_input * -1
            years_employed = days_employed_input / 365
        else:
            days_employed = 365243 # The "unemployed" flag from your notebook
            years_employed = 0
            
        log_income = np.log1p(income)
        
        # --- Set hidden default values ---
        flag_mobil = 1
        flag_work_phone = 0
        flag_phone = 0
        flag_email = 0
        month_from_today = 0
        
        # We don't need 'MONTHS_BALANCE' from the list, as it was part of the 'credit' df,
        # not the 'client' df that forms the basis of the app.
        # Let's rebuild the final feature list from ONLY what we have.
        
        # --- 5. CREATE THE INPUT DATAFRAME ---
        # We will build the dictionary with all our values
        input_dict = {
            'CNT_CHILDREN': children,
            'AMT_INCOME_TOTAL': income,
            'DAYS_BIRTH': days_birth,
            'DAYS_EMPLOYED': days_employed,
            'FLAG_MOBIL': flag_mobil,
            'FLAG_WORK_PHONE': flag_work_phone,
            'FLAG_PHONE': flag_phone,
            'FLAG_EMAIL': flag_email,
            'CNT_FAM_MEMBERS': fam_members,
            'AGE': age,
            'YEARS_EMPLOYED': years_employed,
            'LOG_INCOME': log_income,
            'month_from_today': month_from_today,
            'CODE_GENDER': gender,
            'FLAG_OWN_CAR': car_owner,
            'FLAG_OWN_REALTY': realty_owner,
            'NAME_INCOME_TYPE': income_type,
            'NAME_EDUCATION_TYPE': education_type,
            'NAME_FAMILY_STATUS': family_status,
            'NAME_HOUSING_TYPE': housing_type,
            'OCCUPATION_TYPE': occupation_type
        }
        
        # Create the DataFrame using the *exact* column order from the preprocessor
        input_data = pd.DataFrame(input_dict, index=[0])[ALL_FEATURES]

        with st.spinner("Analyzing applicant..."):
            try:
                # --- 6. PREPROCESS AND PREDICT ---
                input_processed = preprocessor.transform(input_data)
                prediction = model.predict(input_processed)
                probability = model.predict_proba(input_processed)
                
                if prediction[0] == 1: 
                    prob_score = probability[0][1]
                    result_text = "Approved"
                    st.success("🎉 **Result: Approved**")
                    
                else: 
                    prob_score = probability[0][0]
                    result_text = "Rejected"
                    st.error("❌ **Result: Rejected**")

                # --- 7. "FANCY" GAUGE CHART GRAPH ---
                st.subheader(f"Prediction Confidence: {prob_score*100:.2f}%")
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob_score * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"Confidence in {result_text}"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "green" if result_text == "Approved" else "red"},
                        'steps' : [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 100], 'color': "gray"}],
                    }
                ))
                fig.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("What does this graph mean?"):
                    st.write(f"""
                        This gauge shows the model's confidence in its prediction.
                        Based on the data you provided, the model is **{prob_score*100:.2f}%** confident
                        that this applicant should be **{result_text}**.
                    """)

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.exception(e) # This will print the full error details

