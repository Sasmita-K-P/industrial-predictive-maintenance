from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBClassifier


def train_models(
    X_train,
    y_train,
    X_train_scaled
):

    print("\nTraining Logistic Regression...")

    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    lr.fit(
        X_train_scaled,
        y_train
    )

    print("Logistic Regression Training Complete")

    print("\nHyperparameter Tuning Random Forest...")

    rf_param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, 15],
        "min_samples_split": [2, 5]
    }

    rf_grid = GridSearchCV(
        estimator=RandomForestClassifier(
            class_weight="balanced",
            random_state=42
        ),
        param_grid=rf_param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1
    )

    rf_grid.fit(
        X_train,
        y_train
    )

    rf = rf_grid.best_estimator_

    print("Best Random Forest Parameters:")
    print(rf_grid.best_params_)

    print("Random Forest Training Complete")

    print("\nHyperparameter Tuning XGBoost...")

    xgb_params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0]
    }

    xgb_search = RandomizedSearchCV(
        estimator=XGBClassifier(
            random_state=42,
            eval_metric="logloss"
        ),
        param_distributions=xgb_params,
        n_iter=10,
        cv=3,
        scoring="f1",
        n_jobs=-1,
        random_state=42
    )

    xgb_search.fit(
        X_train,
        y_train
    )

    xgb = xgb_search.best_estimator_

    print("Best XGBoost Parameters:")
    print(xgb_search.best_params_)

    print("XGBoost Training Complete")

    return lr, rf, xgb