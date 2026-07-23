import os
import joblib
import optuna
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_recall_curve, auc

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Suppress Optuna verbose logging to keep terminal clean
optuna.logging.set_verbosity(optuna.logging.WARNING)


def calc_f1(y_true, y_pred):
    """Utility to calculate F1 score."""
    return f1_score(y_true, y_pred, zero_division=0)


def tune_logistic_regression(X_train, y_train, X_val, y_val, n_trials=10):
    """Optuna study for Logistic Regression."""
    print("Tuning Logistic Regression with Optuna...")

    def objective(trial):
        c_val = trial.suggest_float("C", 1e-4, 10.0, log=True)
        solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear"])
        
        model = LogisticRegression(
            C=c_val,
            solver=solver,
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return calc_f1(y_val, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_model = LogisticRegression(
        **best_params,
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    best_model.fit(X_train, y_train)
    return best_model, study.best_value


def tune_random_forest(X_train, y_train, X_val, y_val, n_trials=15):
    """Optuna study for Random Forest."""
    print("Tuning Random Forest with Optuna...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1
        }
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return calc_f1(y_val, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_model = RandomForestClassifier(**study.best_params, class_weight="balanced", random_state=42, n_jobs=-1)
    best_model.fit(X_train, y_train)
    return best_model, study.best_value


def tune_xgboost(X_train, y_train, X_val, y_val, n_trials=15):
    """Optuna study for XGBoost."""
    print("Tuning XGBoost with Optuna...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 350, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return calc_f1(y_val, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_model = XGBClassifier(**study.best_params, eval_metric="logloss", random_state=42, n_jobs=-1)
    best_model.fit(X_train, y_train)
    return best_model, study.best_value


def tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=15):
    """Optuna study for LightGBM."""
    print("Tuning LightGBM with Optuna...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 350, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1
        }
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return calc_f1(y_val, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_model = LGBMClassifier(**study.best_params, random_state=42, verbose=-1, n_jobs=-1)
    best_model.fit(X_train, y_train)
    return best_model, study.best_value


def tune_catboost(X_train, y_train, X_val, y_val, n_trials=15):
    """Optuna study for CatBoost."""
    print("Tuning CatBoost with Optuna...")

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 350, step=50),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "random_seed": 42,
            "verbose": 0
        }
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return calc_f1(y_val, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_model = CatBoostClassifier(**study.best_params, random_seed=42, verbose=0)
    best_model.fit(X_train, y_train)
    return best_model, study.best_value


def train_models(X_train, y_train, X_train_scaled, models_dir="models"):
    """
    Trains and tunes Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost.
    Selects the best performing model based on F1-score and saves it.
    """
    os.makedirs(models_dir, exist_ok=True)

    # Use internal validation split (80/20 of training set) for Optuna objective evaluation
    split_idx = int(len(X_train) * 0.8)
    
    X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]
    
    X_tr_sc, X_val_sc = X_train_scaled[:split_idx], X_train_scaled[split_idx:]

    print("\n================ STARTING OPTUNA BENCHMARK ================\n")

    # Tune all 5 architectures
    lr_model, lr_score = tune_logistic_regression(X_tr_sc, y_tr, X_val_sc, y_val)
    rf_model, rf_score = tune_random_forest(X_tr, y_tr, X_val, y_val)
    xgb_model, xgb_score = tune_xgboost(X_tr, y_tr, X_val, y_val)
    lgbm_model, lgbm_score = tune_lightgbm(X_tr, y_tr, X_val, y_val)
    cat_model, cat_score = tune_catboost(X_tr, y_tr, X_val, y_val)

    # Leaderboard Summary
    results = {
        "Logistic Regression": (lr_model, lr_score),
        "Random Forest": (rf_model, rf_score),
        "XGBoost": (xgb_model, xgb_score),
        "LightGBM": (lgbm_model, lgbm_score),
        "CatBoost": (cat_model, cat_score)
    }

    print("\n================ OPTUNA LEADERBOARD (Val F1) ================")
    for name, (_, score) in results.items():
        print(f"{name:<22} : F1 Score = {score:.4f}")
    print("============================================================\n")

    # Determine Best Overall Model
    best_model_name = max(results, key=lambda k: results[k][1])
    best_model, best_f1 = results[best_model_name]

    print(f"🏆 Champion Model: {best_model_name} with F1 = {best_f1:.4f}\n")

    # Save artifacts
    joblib.dump(best_model, os.path.join(models_dir, "best_model.pkl"))
    joblib.dump(xgb_model, os.path.join(models_dir, "xgb_model.pkl")) # Backward compatibility

    return lr_model, rf_model, xgb_model, lgbm_model, cat_model, best_model, best_model_name