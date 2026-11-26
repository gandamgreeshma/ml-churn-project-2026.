
import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

# Application title
st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer details to predict whether "
    "the customer is likely to leave the company."
)

st.divider()

# Input form
st.header("Customer Information")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )
        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )
        security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )
        backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

    with col2:
        protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )
        support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )
        tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )
        movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0
        )
        total = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0
        )

    submitted = st.form_submit_button(
        "Predict Customer Churn",
        use_container_width=True
    )

# Prediction
if submitted:

    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()
    st.header("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn!")
    else:
        st.success("✅ Customer is likely to stay!")

    st.metric(
        "Churn Probability",
        f"{probability:.2%}"
    )

    st.progress(float(probability))
