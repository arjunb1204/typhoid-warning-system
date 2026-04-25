import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Load trained model + preprocessors
# -------------------------------
rf_model = joblib.load("rf_outbreak_model.joblib")
scaler = joblib.load("scaler.joblib")
ohe = joblib.load("onehot_encoder.joblib")

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Disease Outbreak Early Warning System", page_icon="🦠", layout="wide")

st.title("🦠 Disease Outbreak Early Warning System")
st.markdown(
    """
    <style>
    .big-font {font-size:22px !important;}
    </style>
    """, unsafe_allow_html=True
)

st.markdown(
    """
    ### Welcome to the Outbreak Predictor Dashboard  
    This system uses **AI & clinical/environmental data** to **predict disease outbreak risks** in local communities.  
    Adjust the parameters in the sidebar and see outbreak risk levels, **trends**, and **key drivers** of outbreaks.  
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("⚙️ Simulation Settings")

week = st.sidebar.slider("Week", 1, 52, 12)
total_patients = st.sidebar.number_input("Total Patients", min_value=10, max_value=500, value=100, step=10)
avg_fever_duration = st.sidebar.slider("Avg Fever Duration (Days)", 0, 15, 5)
avg_wbc = st.sidebar.number_input("Avg WBC Count", min_value=3000, max_value=20000, value=7000, step=100)
avg_platelets = st.sidebar.number_input("Avg Platelet Count", min_value=50000, max_value=450000, value=250000, step=5000)
unsafe_water_pct = st.sidebar.slider("Unsafe Water (%)", 0, 100, 30)
poor_hygiene_pct = st.sidebar.slider("Poor Hygiene (%)", 0, 100, 20)
streetfood_pct = st.sidebar.slider("Street Food Consumption (%)", 0, 100, 40)
vaccinated_pct = st.sidebar.slider("Vaccinated (%)", 0, 100, 10)
ongoing_infection_pct = st.sidebar.slider("Ongoing Infection in Society (%)", 0, 100, 50)
weather = st.sidebar.selectbox("Weather Condition", ["Rainy", "Dry", "Humid"])
location = st.sidebar.text_input("City", "Delhi")
sublocality = st.sidebar.text_input("SubLocality", "Delhi-North")

# -------------------------------
# Data Preparation
# -------------------------------
num_features = pd.DataFrame([[
    total_patients, avg_fever_duration, avg_wbc, avg_platelets,
    unsafe_water_pct, poor_hygiene_pct, streetfood_pct,
    vaccinated_pct, ongoing_infection_pct, 0, week
]], columns=[
    "total_patients","avg_fever_duration","avg_wbc","avg_platelets",
    "unsafe_water_pct","poor_hygiene_pct","streetfood_pct",
    "vaccinated_pct","ongoing_infection_pct","typhoid_rate_pct","Week"
])

cat_features = pd.DataFrame([[weather, location, sublocality]],
                            columns=["Weather Condition","Location","SubLocality"])
cat_encoded = ohe.transform(cat_features.astype(str))
cat_df = pd.DataFrame(cat_encoded, columns=ohe.get_feature_names_out())

# Scale numeric
num_scaled = scaler.transform(num_features[num_features.columns])
X_final = np.hstack([num_scaled, cat_df.values])

# -------------------------------
# Prediction
# -------------------------------
outbreak_prob = rf_model.predict_proba(X_final)[:,1][0]
risk_level = "Low"
color = "green"
if outbreak_prob > 0.7:
    risk_level, color = "High", "red"
elif outbreak_prob > 0.4:
    risk_level, color = "Medium", "orange"

# -------------------------------
# Layout (3 Columns)
# -------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="📊 Outbreak Probability", value=f"{outbreak_prob*100:.2f}%")

with col2:
    st.metric(label="⚠️ Risk Level", value=risk_level)

with col3:
    st.metric(label="📅 Week", value=week)

st.markdown(f"<p class='big-font' style='color:{color};'>Predicted Risk Level: {risk_level} ({outbreak_prob*100:.1f}%)</p>", unsafe_allow_html=True)

# -------------------------------
# Trend Over Weeks (with risk bands)
# -------------------------------
st.markdown("### 📈 Outbreak Risk Trend (Simulated Over Weeks)")

weeks = list(range(1, 53))
probs = []
for w in weeks:
    temp_num = num_features.copy()
    temp_num["Week"] = w
    temp_scaled = scaler.transform(temp_num[temp_num.columns])
    X_temp = np.hstack([temp_scaled, cat_df.values])
    prob = rf_model.predict_proba(X_temp)[:,1][0]
    probs.append(prob)

trend_df = pd.DataFrame({"Week": weeks, "Outbreak Probability": probs})

# Custom matplotlib chart with shaded bands
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(trend_df["Week"], trend_df["Outbreak Probability"], color="blue", linewidth=2)

# Shaded regions for Low/Medium/High
ax.axhspan(0, 0.4, facecolor="green", alpha=0.1)
ax.axhspan(0.4, 0.7, facecolor="orange", alpha=0.1)
ax.axhspan(0.7, 1.0, facecolor="red", alpha=0.1)

ax.set_title("Outbreak Probability Trend Over Weeks")
ax.set_xlabel("Week")
ax.set_ylabel("Outbreak Probability")
ax.set_ylim(0,1)
st.pyplot(fig)

# -------------------------------
# Feature Importance (Top Drivers)
# -------------------------------
st.markdown("### 🔑 Top Factors Driving Outbreak Prediction")

feature_names = list(num_features.columns) + list(cat_df.columns)
importances = rf_model.feature_importances_
feat_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
feat_imp = feat_imp.sort_values("importance", ascending=False).head(5)

st.bar_chart(feat_imp.set_index("feature"))

# -------------------------------
# Input Recap
# -------------------------------
st.markdown("### 📝 Simulation Input Recap")
st.dataframe(num_features.join(cat_features))
