import joblib

from data_preprocessing import preprocess_data
from train import train_models
from evaluate import evaluate_model
from explainability import explain_model


DATA_PATH = "data/raw/ai4i2020.csv"

(
    X_train,
    X_test,
    y_train,
    y_test,
    X_train_scaled,
    X_test_scaled,
    scaler,
    feature_names
) = preprocess_data(DATA_PATH)

lr, rf, xgb = train_models(
    X_train,
    y_train,
    X_train_scaled
)

evaluate_model(
    lr,
    X_test_scaled,
    y_test,
    "Logistic Regression"
)

evaluate_model(
    rf,
    X_test,
    y_test,
    "Random Forest"
)

evaluate_model(
    xgb,
    X_test,
    y_test,
    "XGBoost"
)

explain_model(
    xgb,
    feature_names
)

joblib.dump(
    xgb,
    "models/xgb_model.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("\nModel Saved Successfully")