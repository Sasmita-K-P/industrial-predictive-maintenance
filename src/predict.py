import joblib
import pandas as pd
from data_preprocessing import create_domain_features

# 1. Load Production Artifacts
try:
    model = joblib.load("models/best_model.pkl")
    print("Loaded Champion Model (best_model.pkl)")
except FileNotFoundError:
    model = joblib.load("models/xgb_model.pkl")
    print("Fallback to XGBoost Model (xgb_model.pkl)")

type_encoder = joblib.load("models/type_encoder.pkl")
feature_selector = joblib.load("models/feature_selector.pkl")
selected_features = joblib.load("models/selected_features.pkl")
scaler = joblib.load("models/scaler.pkl")

# 2. Raw Simulated Incoming Telemetry Data (Raw format)
raw_telemetry = pd.DataFrame({
    "Type": ["M"],
    "Air_temperature_K": [300.0],
    "Process_temperature_K": [310.0],
    "Rotational_speed_rpm": [1350],
    "Torque_Nm": [55.0],
    "Tool_wear_min": [210]
})

# 3. Preprocessing & Feature Pipeline for Inference
df_inference = raw_telemetry.copy()

# Encode Categorical Type
df_inference["Type"] = type_encoder.transform(df_inference["Type"])

# Apply Domain Physical Feature Engineering
df_inference = create_domain_features(df_inference)

# Apply Feature Selection (Ensures exact k features are selected)
X_selected = feature_selector.transform(df_inference)
X_inference = pd.DataFrame(X_selected, columns=selected_features)

# Handle scaling if model is Logistic Regression
if type(model).__name__ == "LogisticRegression":
    X_inference = scaler.transform(X_inference)

# 4. Predict Outcome & Failure Probability
prediction = model.predict(X_inference)[0]

if hasattr(model, "predict_proba"):
    probability = model.predict_proba(X_inference)[0][1]
    print(f"\nFailure Probability: {probability:.2%}")

print("\n================ INFERENCE RESULT ================")
if prediction == 1:
    print("⚠ CRITICAL ALERT: High Failure Likelihood Detected!")
else:
    print("✅ SYSTEM NORMAL: Equipment Operating Within Parameters.")
print("==================================================\n")