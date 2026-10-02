import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# =========================
# LOAD MODEL + PREPROCESSOR
# =========================
model = joblib.load("app/model.pkl")
preprocessor = joblib.load("app/preprocessor.pkl")

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.main { background-color: #0E1117; }

h1 {
    color: white;
    text-align: center;
    font-size: 42px;
}

h3 { color: white; }

.stButton>button {
    width: 100%;
    background: linear-gradient(to right, #4CAF50, #45a049);
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 3em;
    border: none;
}

.metric-card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("<h3 style='text-align: center;'>AI-powered churn risk analysis system</h3>", unsafe_allow_html=True)

st.write("")

# =========================
# SIMPLIFIED USER INPUTS
# =========================
st.sidebar.header("Customer Details (Simple Mode)")

tenure = st.sidebar.slider("How long has the customer stayed? (Months)", 0, 72, 12)

monthly_charges = st.sidebar.number_input("Monthly Bill Amount ($)", 0.0, 200.0, 70.0)

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

payment_method = st.sidebar.selectbox(
    "Preferred Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

# =========================
# PREDICT BUTTON
# =========================
predict_button = st.button("🔮 Predict Churn Risk")

# =========================
# PREDICTION LOGIC
# =========================
if predict_button:

    # =========================
    # AUTO-FILLED TECH FEATURES
    # (Hidden from user)
    # =========================
    input_data = pd.DataFrame([{
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": contract,
        "PaperlessBilling": "Yes",
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": float(tenure * monthly_charges)
    }])

    # =========================
    # PREPROCESS
    # =========================
    processed_data = preprocessor.transform(input_data)

    # =========================
    # PREDICTION
    # =========================
    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0][1]

    # =========================
    # UI OUTPUT
    # =========================
    st.subheader("Prediction Confidence")
    st.progress(float(probability))

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Churn Probability</h3>
            <h1>{probability:.2%}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        status = "⚠️ Likely to Churn" if prediction == 1 else "✅ Likely to Stay"

        st.markdown(f"""
        <div class="metric-card">
            <h3>Prediction</h3>
            <h1>{status}</h1>
        </div>
        """, unsafe_allow_html=True)

    # =========================
    # RISK LEVEL
    # =========================
    if probability > 0.7:
        st.error("🚨 High Risk Customer")
    elif probability > 0.4:
        st.warning("⚠️ Medium Risk Customer")
    else:
        st.success("✅ Low Risk Customer")

    # =========================
    # CHART
    # =========================
    st.subheader("Probability Breakdown")

    labels = ["Stay", "Churn"]
    values = [1 - probability, probability]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel("Probability")
    ax.set_title("Churn vs Stay")

    st.pyplot(fig)

    # =========================
    # FEATURE IMPORTANCE
    # =========================
    # st.subheader("Top Influencing Factors")

    # try:
    #     feature_names = preprocessor.get_feature_names_out()
    #     importances = model.feature_importances_

    #     indices = np.argsort(importances)[-10:]

    #     top_features = np.array(feature_names)[indices]
    #     top_importances = importances[indices]

    #     fig2, ax2 = plt.subplots()
    #     ax2.barh(top_features, top_importances)
    #     ax2.set_title("Top 10 Features")

    #     st.pyplot(fig2)

    # except:
    #     st.info("Feature importance not available for this model.")