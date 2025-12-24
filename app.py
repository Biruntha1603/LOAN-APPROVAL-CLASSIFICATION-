import streamlit as st
import joblib
import numpy as np

# Load your KNN model
model = joblib.load('knn_model.pkl')

# Title
st.title("Loan Approval Classification")

# Input form (you can customize based on your dataset features)
LoanID = st.number_input("Loan_ID",min_value=0,max_value=5000,value=7)
Gender = st.radio("Gender", ["Male","Female"])
Married= st.selectbox("Married", ["Yes","No"])
Dependents = st.number_input("Dependents", min_value=0,max_value=5,value=3)
Education = st.radio("Education", ["Graduate","Not graduate"])
SelfEmployed = st.selectbox("Self_Employed", ["Yes","No"])
ApplicantIncome = st.number_input("ApplicantIncome", min_value=0,max_value=100000,value=4000)
CoapplicantIncome= st.number_input("CoapplicantIncome",min_value=0,max_value=100000, value=2000)
LoanAmount = st.number_input("LoanAmount", min_value=0,max_value=100000,value=500)
CreditScore = st.radio("Credit_Score", ["0","1"])
PropertyArea = st.selectbox("Property_Area", ["Urban","Rural","Semiurban"])

# --- Map categorical values to numeric (must match your training encoding) ---
gender_map = {"Male": 0, "Female": 1}
married_map = {"Yes": 1, "No": 0}
education_map = {"Not graduate": 0, "Graduate": 1}
selfemployed_map={"Yes": 1, "No": 0}
propertyarea_map = {"Urban":0,"Rural":1,"Semiurban":2}
# Encode inputs
input_features = [
    LoanID,
    gender_map[Gender],
    married_map[Married],
    Dependents,
    education_map[Education],
    selfemployed_map[SelfEmployed],
    ApplicantIncome,
    CoapplicantIncome,
    LoanAmount,
    CreditScore,
    propertyarea_map[PropertyArea]
]

input_array = np.array([input_features],dtype=float)


# --- Predict ---
if st.button("Predict Approved or Not"):
    prediction = model.predict(input_array)[0]
    st.success(f"Prediction: {'Approved' if prediction == 1 else 'Not Approved'}")

