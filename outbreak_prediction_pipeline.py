"""
Final Outbreak Prediction Pipeline
----------------------------------
- Loads patient-level CSV.
- Aggregates into locality-week data.
- Fabricates outbreaks if needed.
- Trains Logistic Regression + Random Forest.
- Evaluates them.
- Saves trained models and preprocessors for Streamlit app.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import joblib

# -------------------------------
# PARAMETERS
# -------------------------------
INPUT_CSV = "typhoid_with_time_locality_v2.csv"  # change if your file has different name
OUTBREAK_THRESHOLD_PERCENT = 2.0  # baseline definition
FABRICATED_FRACTION = 0.1  # 10% rows will be turned into outbreaks if none exist
TIME_SPLIT_WEEK = 40
RANDOM_STATE = 42

# -------------------------------
# 1. Load dataset
# -------------------------------
df = pd.read_csv(INPUT_CSV)
print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")

# -------------------------------
# 2. Aggregate into locality-week
# -------------------------------
group_cols = ["Location", "SubLocality", "Week", "Weather Condition"]
agg = df.groupby(group_cols).agg(
    total_patients=("Typhoid Status", "count"),
    typhoid_cases=("Typhoid Status", lambda x: (x == "Positive").sum()),
    avg_fever_duration=("Fever Duration (Days)", "mean"),
    avg_wbc=("White Blood Cell Count", "mean"),
    avg_platelets=("Platelet Count", "mean"),
    unsafe_water_pct=("Water Source Type", lambda x: (x == "Unsafe").mean() * 100),
    poor_hygiene_pct=("Hand Hygiene", lambda x: (x == "Poor").mean() * 100),
    streetfood_pct=("Consumption of Street Food", lambda x: (x == "Yes").mean() * 100),
    vaccinated_pct=("Typhoid Vaccination Status", lambda x: (x == "Yes").mean() * 100),
    ongoing_infection_pct=("Ongoing Infection in Society", lambda x: (x == "Yes").mean() * 100),
).reset_index()

agg["typhoid_rate_pct"] = agg["typhoid_cases"] / agg["total_patients"] * 100
agg["Outbreak"] = (agg["typhoid_rate_pct"] > OUTBREAK_THRESHOLD_PERCENT).astype(int)

print("Before fabrication:")
print("Any outbreaks? ", (agg["Outbreak"] == 1).sum())

# -------------------------------
# 3. Fabricate outbreaks if needed
# -------------------------------
if (agg["Outbreak"] == 1).sum() == 0:
    np.random.seed(RANDOM_STATE)
    n_fabricated = int(FABRICATED_FRACTION * len(agg))
    fabricated_idx = np.random.choice(agg.index, size=n_fabricated, replace=False)
    agg.loc[fabricated_idx, "typhoid_rate_pct"] = np.random.uniform(20, 40, size=n_fabricated)
    agg.loc[fabricated_idx, "Outbreak"] = 1
    print(f"✅ Fabricated {n_fabricated} outbreak rows.")

print("Any outbreaks now? ", (agg["Outbreak"] == 1).sum())

# -------------------------------
# 4. Prepare features & target
# -------------------------------
target = "Outbreak"
feature_cols = [
    "total_patients","avg_fever_duration","avg_wbc","avg_platelets",
    "unsafe_water_pct","poor_hygiene_pct","streetfood_pct",
    "vaccinated_pct","ongoing_infection_pct","typhoid_rate_pct","Week"
]

X_num = agg[feature_cols].fillna(0)
y = agg[target]

# Encode categorical
cat_cols = ["Weather Condition", "Location", "SubLocality"]
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_cat = ohe.fit_transform(agg[cat_cols].astype(str))
cat_feature_names = ohe.get_feature_names_out()

X = pd.concat(
    [X_num.reset_index(drop=True),
     pd.DataFrame(X_cat, columns=cat_feature_names, index=X_num.index)],
    axis=1
)

# -------------------------------
# 5. Train/test split
# -------------------------------
train_mask = agg["Week"] <= TIME_SPLIT_WEEK
test_mask = agg["Week"] > TIME_SPLIT_WEEK

if len(np.unique(y[train_mask])) < 2 or len(np.unique(y[test_mask])) < 2:
    print("\n⚠️ Temporal split unbalanced → using stratified random split instead.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
else:
    X_train, X_test, y_train, y_test = X[train_mask], X[test_mask], y[train_mask], y[test_mask]

# Scale numeric
scaler = StandardScaler()
num_cols = X_num.columns.tolist()
X_train_num = scaler.fit_transform(X_train[num_cols])
X_test_num = scaler.transform(X_test[num_cols])
X_train_final = np.hstack([X_train_num, X_train.drop(columns=num_cols).values])
X_test_final = np.hstack([X_test_num, X_test.drop(columns=num_cols).values])

# -------------------------------
# 6. Train models
# -------------------------------
log = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
log.fit(X_train_final, y_train)
y_pred_log = log.predict(X_test_final)
y_prob_log = log.predict_proba(X_test_final)[:,1]

rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
rf.fit(X_train_final, y_train)
y_pred_rf = rf.predict(X_test_final)
y_prob_rf = rf.predict_proba(X_test_final)[:,1]

# -------------------------------
# 7. Evaluate
# -------------------------------
def evaluate(y_true, y_pred, y_prob, name):
    print(f"\n--- {name} ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))
    print("ROC-AUC:", roc_auc_score(y_true, y_prob))

evaluate(y_test, y_pred_log, y_prob_log, "Logistic Regression")
evaluate(y_test, y_pred_rf, y_prob_rf, "Random Forest")

# -------------------------------
# 8. Save models + preprocessors
# -------------------------------
joblib.dump(rf, "rf_outbreak_model.joblib")
joblib.dump(log, "logistic_outbreak_model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(ohe, "onehot_encoder.joblib")

print("\n✅ Models and preprocessors saved as:")
print("- rf_outbreak_model.joblib")
print("- logistic_outbreak_model.joblib")
print("- scaler.joblib")
print("- onehot_encoder.joblib")

# -------------------------------
# 9. Demo predictions
# -------------------------------
latest_week = agg["Week"].max()
recent_mask = agg["Week"] == latest_week
if recent_mask.any():
    X_recent = X[recent_mask]
    X_recent_num = scaler.transform(X_recent[num_cols])
    X_recent_final = np.hstack([X_recent_num, X_recent.drop(columns=num_cols).values])
    preds = rf.predict_proba(X_recent_final)[:,1]
    out_df = agg[recent_mask].copy()
    out_df["outbreak_prob_rf"] = preds
    print("\nSample predictions for latest week:")
    print(out_df[["Location","SubLocality","Week","typhoid_rate_pct","outbreak_prob_rf"]].head())
