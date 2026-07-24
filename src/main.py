import joblib
import pandas as pd
import mlflow

from data_preprocessing import preprocess_data
from train import train_models, sanitize_df
from evaluate import evaluate_model
from explainability import explain_model


DATA_PATH = "data/raw/ai4i2020.csv"

def main():
    print("==================================================")
    print("   PREDICTIVE MAINTENANCE MLOPS PIPELINE RUNNING   ")
    print("==================================================\n")

    # Step 1: Preprocessing & Feature Engineering
    print("[1/4] Running Data Preprocessing & Physical Feature Engineering...")
    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        scaler,
        feature_names
    ) = preprocess_data(DATA_PATH, use_smote=True, k_features=8)

    # Sanitize feature names for XGBoost / LightGBM compatibility
    X_test_clean = sanitize_df(X_test)

    # Step 2: Optuna Benchmark & Model Training
    print("\n[2/4] Tuning & Training Models via Optuna...")
    (
        lr,
        rf,
        xgb,
        lgbm,
        cat,
        best_model,
        best_model_name
    ) = train_models(
        X_train,
        y_train,
        X_train_scaled
    )

    # Step 3: Evaluate Models on Unseen Test Data
    print("\n[3/4] Evaluating Models on Test Split...")

    # Pass sanitized DataFrames where appropriate and unpack the metrics dictionary
    lr_metrics, _ = evaluate_model(lr, X_test_scaled, y_test, "Logistic Regression")
    rf_metrics, _ = evaluate_model(rf, X_test, y_test, "Random Forest")
    xgb_metrics, _ = evaluate_model(xgb, X_test_clean, y_test, "XGBoost")
    lgbm_metrics, _ = evaluate_model(lgbm, X_test_clean, y_test, "LightGBM")
    cat_metrics, _ = evaluate_model(cat, X_test, y_test, "CatBoost")

    # Test Results Leaderboard Printout
    results = [lr_metrics, rf_metrics, xgb_metrics, lgbm_metrics, cat_metrics]
    summary_df = pd.DataFrame(results)

    print("\n==================================================")
    print("         FINAL TEST SET LEADERBOARD               ")
    print("==================================================")
    print(summary_df.to_string(index=False))
    print("==================================================\n")

    # Log Final Test Evaluation Summary to MLflow
    with mlflow.start_run(run_name="Final_Test_Set_Evaluation"):
        for metric_dict in results:
            model_name = metric_dict.get("Model", "Unknown").replace(" ", "_")
            for k, v in metric_dict.items():
                if k != "Model" and isinstance(v, (int, float)):
                    mlflow.log_metric(f"{model_name}_test_{k.lower()}", v)

    # Step 4: Model Explainability & Artifact Saving
    print(f"[4/4] Generating Feature Importance & SHAP Plots for Champion ({best_model_name})...")
    
    # Check if champion is Logistic Regression to pass scaled data, otherwise pass unscaled X_test
    if best_model_name == "Logistic Regression":
        X_explain = X_test_scaled
    elif best_model_name in ["XGBoost", "LightGBM"]:
        X_explain = X_test_clean
    else:
        X_explain = X_test

    explain_model(best_model, feature_names, X_test=X_explain)

    # Explicit Model Artifact Persistence
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(xgb, "models/xgb_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("\n==================================================")
    print("  PIPELINE EXECUTED SUCCESSFULLY - ALL ARTIFACTS SAVED ")
    print("==================================================")


if __name__ == "__main__":
    main()