# 🦠 Disease Outbreak Early Warning & Awareness System

**Tagline:** Predicting Typhoid Outbreaks with Data: Saving Lives Before They're at Risk.

## 📌 Project Overview
The **Disease Outbreak Early Warning System** is a data-driven application designed to predict the probability of a typhoid outbreak in a specific location by analyzing environmental, clinical, and social factors.

Current public health systems often react to outbreaks only *after* significant transmission has occurred. This project aims to shift the paradigm from reactive to proactive, providing local health authorities with actionable, real-time insights to launch targeted awareness campaigns, sanitation drives, and vaccination clinics *before* an outbreak peaks.

## ✨ Key Features
- **Predictive Machine Learning Models:** Utilizes Random Forest and Logistic Regression classification models to evaluate the risk levels of an outbreak based on patient characteristics and environmental factors.
- **Interactive Dashboard:** A Streamlit-based web application providing a user-friendly interface to simulate risks, view simulated trends over weeks, and identify the top factors driving a potential outbreak.
- **Comprehensive Feature Set:** Factors in variables like Average Fever Duration, Platelet Count, Unsafe Water percentage, Poor Hygiene, Street Food Consumption, and Vaccination Status.
- **Data Pipeline:** End-to-end data processing script that aggregates patient-level records into locality-week data and prepares it for predictive modeling.

## 🛠️ Tech Stack
- **Data Analysis & ML:** Python (Pandas, NumPy, Scikit-learn)
- **Web App / Dashboard:** Streamlit
- **Visualization:** Matplotlib, Seaborn
- **Model Serialization:** Joblib

## 📂 Project Structure
- `app.py`: The main Streamlit dashboard application displaying the predicted risk level, outbreak trends, and key drivers.
- `outbreak_prediction_pipeline.py`: The data pipeline script that curates data, handles feature engineering, trains the models (Logistic Regression, Random Forest), and saves them.
- `eda_typhoid.py`: Script dedicated to exploratory data analysis (EDA) of the typhoid patient datasets.
- `fabricate_outbreak.py` & `age_vs_fever_duration.py`: Auxiliary scripts for data manipulation and specific analysis.
- `*.joblib`: Serialized pre-trained machine learning models and data preprocessors.
- `*.csv`: Dataset files containing patient and localized metadata for training and prediction.

## 🚀 Getting Started

### Prerequisites
Make sure you have Python installed. You will need the following libraries:
```bash
pip install pandas numpy scikit-learn streamlit matplotlib joblib
```

### Running the Application
1. **Model Training (Optional):**
   If you wish to retrain the machine learning models on a new dataset, run the prediction pipeline:
   ```bash
   python outbreak_prediction_pipeline.py
   ```
   *Note: This will output `rf_outbreak_model.joblib`, `logistic_outbreak_model.joblib`, `scaler.joblib`, and `onehot_encoder.joblib`.*

2. **Starting the Dashboard:**
   Run the Streamlit application using the following command:
   ```bash
   streamlit run app.py
   ```
   The dashboard should open automatically in your default internet browser.

## 🔮 Future Work
- Integration with live data feeds from local health clinics and weather APIs.
- Expanding the scope to predict other water-borne diseases like Cholera and Dysentery.
- Developing an SMS or WhatsApp-based automated alert system for citizens in high-risk zones.

## 🤝 Social Impact
By shifting from reactive metrics to proactive intelligence, this tool helps optimize the allocation of medical supplies, vaccines, and personnel, effectively empowering communities and protecting vulnerable populations.
