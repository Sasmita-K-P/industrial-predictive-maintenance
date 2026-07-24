import os
import re
import joblib
import optuna
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import xgboost as xgb  
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost

# Connect to your local MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Create your project experiment
mlflow.set_experiment("Predictive_Maintenance_Models")

# Import data preprocessing function
from data_preprocessing import preprocess_data

# Suppress Optuna verbosity for clean logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def sanitize_df(df):
    """Sanitizes DataFrame columns for XGBoost and LightGBM compatibility."""
    if isinstance(df, pd.DataFrame):
        clean_cols = [
            re.sub(r"[\[\]<>]", "", c).replace(" ", "_") for c in df.columns
        ]
        df = df.copy()
        df.columns = clean_cols
    return df


def eval_metrics(y_true, y_pred):
    """Calculates comprehensive classification metrics for predictive maintenance."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def tune_logistic_regression(X_train, y_train, X_val, y_val, n_trials=10):
    def objective(trial):
        C = trial.suggest_float("C", 1e-4, 10.0, log=True)
        solver = trial.suggest_categorical(
            "solver", ["liblinear", "lbfgs", "saga"]
        )
        model = LogisticRegression(
            C=C, solver=solver, max_iter=1000, random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return f1_score(y_val, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_model = LogisticRegression(
        **study.best_params, max_iter=1000, random_state=42
    )
    best_model.fit(X_train, y_train)
    return best_model, study.best_value, study.best_params


def tune_random_forest(X_train, y_train, X_val, y_val, n_trials=10):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "random_state": 42,
        }
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return f1_score(y_val, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_model = RandomForestClassifier(**study.best_params, random_state=42)
    best_model.fit(X_train, y_train)
    return best_model, study.best_value, study.best_params


def tune_xgboost(X_train, y_train, X_val, y_val, n_trials=10):
    X_train = sanitize_df(X_train)
    X_val = sanitize_df(X_val)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            "random_state": 42,
            "eval_metric": "logloss",
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return f1_score(y_val, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_model = xgb.XGBClassifier(
        **study.best_params, random_state=42, eval_metric="logloss"
    )
    best_model.fit(X_train, y_train)
    return best_model, study.best_value, study.best_params


def tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=10):
    X_train = sanitize_df(X_train)
    X_val = sanitize_df(X_val)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "random_state": 42,
            "verbose": -1,
        }
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return f1_score(y_val, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_model = LGBMClassifier(
        **study.best_params, random_state=42, verbose=-1
    )
    best_model.fit(X_train, y_train)
    return best_model, study.best_value, study.best_params


def tune_catboost(X_train, y_train, X_val, y_val, n_trials=10):
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 50, 300),
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "random_state": 42,
            "verbose": 0,
        }
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return f1_score(y_val, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_model = CatBoostClassifier(
        **study.best_params, random_state=42, verbose=0
    )
    best_model.fit(X_train, y_train)
    return best_model, study.best_value, study.best_params


def train_models(X_train, y_train, X_train_scaled):
    # Train/Val split for hyperparameter optimization
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    X_tr_s, X_val_s, _, _ = train_test_split(
        X_train_scaled,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    print("\n================ STARTING OPTUNA BENCHMARK ================\n")

    # Master Parent Run in MLflow
    with mlflow.start_run(run_name="Benchmark_Optuna_Comparison") as parent_run:
        mlflow.log_param("num_train_samples", X_train.shape[0])
        mlflow.log_param("num_features", X_train.shape[1])

        # ---------------- 1. Logistic Regression ----------------
        print("Tuning Logistic Regression with Optuna...")
        lr, lr_score, lr_params = tune_logistic_regression(X_tr_s, y_tr, X_val_s, y_val)
        lr_val_preds = lr.predict(X_val_s)
        lr_metrics = eval_metrics(y_val, lr_val_preds)
        
        with mlflow.start_run(run_name="Logistic_Regression", nested=True):
            mlflow.log_params(lr_params)
            mlflow.log_metrics(lr_metrics)
            mlflow.sklearn.log_model(lr, artifact_path="model")

        # ---------------- 2. Random Forest ----------------
        print("Tuning Random Forest with Optuna...")
        rf, rf_score, rf_params = tune_random_forest(X_tr, y_tr, X_val, y_val)
        rf_val_preds = rf.predict(X_val)
        rf_metrics = eval_metrics(y_val, rf_val_preds)
        
        with mlflow.start_run(run_name="Random_Forest", nested=True):
            mlflow.log_params(rf_params)
            mlflow.log_metrics(rf_metrics)
            mlflow.sklearn.log_model(rf, artifact_path="model")

        # ---------------- 3. XGBoost ----------------
        print("Tuning XGBoost with Optuna...")
        xgb_m, xgb_score, xgb_params = tune_xgboost(X_tr, y_tr, X_val, y_val)
        xgb_val_preds = xgb_m.predict(sanitize_df(X_val))
        xgb_metrics = eval_metrics(y_val, xgb_val_preds)
        
        with mlflow.start_run(run_name="XGBoost", nested=True):
            mlflow.log_params(xgb_params)
            mlflow.log_metrics(xgb_metrics)
            mlflow.xgboost.log_model(xgb_m, artifact_path="model")

        # ---------------- 4. LightGBM ----------------
        print("Tuning LightGBM with Optuna...")
        lgbm, lgbm_score, lgbm_params = tune_lightgbm(X_tr, y_tr, X_val, y_val)
        lgbm_val_preds = lgbm.predict(sanitize_df(X_val))
        lgbm_metrics = eval_metrics(y_val, lgbm_val_preds)
        
        with mlflow.start_run(run_name="LightGBM", nested=True):
            mlflow.log_params(lgbm_params)
            mlflow.log_metrics(lgbm_metrics)
            mlflow.lightgbm.log_model(lgbm, artifact_path="model")

        # ---------------- 5. CatBoost ----------------
        print("Tuning CatBoost with Optuna...")
        cat, cat_score, cat_params = tune_catboost(X_tr, y_tr, X_val, y_val)
        cat_val_preds = cat.predict(X_val)
        cat_metrics = eval_metrics(y_val, cat_val_preds)
        
        with mlflow.start_run(run_name="CatBoost", nested=True):
            mlflow.log_params(cat_params)
            mlflow.log_metrics(cat_metrics)
            mlflow.catboost.log_model(cat, artifact_path="model")

        scores = {
            "Logistic Regression": (lr, lr_score),
            "Random Forest": (rf, rf_score),
            "XGBoost": (xgb_m, xgb_score),
            "LightGBM": (lgbm, lgbm_score),
            "CatBoost": (cat, cat_score),
        }

        print("\n--- OPTUNA BENCHMARK RESULTS (Validation F1 Score) ---")
        for name, (_, score) in scores.items():
            print(f"  * {name}: {score:.4f}")

        # Pick the champion model with the highest validation F1
        best_model_name, (best_model, best_f1) = max(
            scores.items(), key=lambda item: item[1][1]
        )
        print(
            f"\n🏆 Best Model Selected: {best_model_name} (Val F1: {best_f1:.4f})"
        )

        # Log champion model info to the parent MLflow run
        mlflow.log_param("champion_model_name", best_model_name)
        mlflow.log_metric("champion_val_f1", best_f1)

    return lr, rf, xgb_m, lgbm, cat, best_model, best_model_name


if __name__ == "__main__":
    # Load raw data and preprocess
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, feature_names = preprocess_data(
        file_path="data/raw/ai4i2020.csv", use_smote=True, k_features=8
    )

    # Train and select champion model
    lr, rf, xgb_m, lgbm, cat, best_model, best_model_name = train_models(
        X_train, y_train, X_train_scaled
    )

    # Save outputs to models/
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(xgb_m, "models/xgb_model.pkl")
    print("\nSaved models/best_model.pkl and models/xgb_model.pkl")