import os
import re
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def clean_column_names(df):
    """Sanitizes DataFrame column names for XGBoost & LightGBM compatibility.
    
    Removes forbidden characters like [, ], <, >, and replaces spaces with underscores.
    """
    clean_cols = []
    for col in df.columns:
        col = re.sub(r'[\[\]<>]', '', col)  # Remove [, ], <, >
        col = re.sub(r'[\s/]+', '_', col)    # Replace spaces & slashes with underscores
        clean_cols.append(col)
    df.columns = clean_cols
    return df


def preprocess_data(
    file_path="data/raw/ai4i2020.csv",
    use_smote=True,
    k_features=8,
    save_processed=True,
):
    """Preprocesses raw industrial machine telemetry data.

    Engineers domain physics features, cleans column names, scales inputs,
    selects K best features, balances classes with SMOTE, and saves datasets.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    # 1. Load Raw Dataset
    df = pd.read_csv(file_path)

    # 2. Clean Column Names for XGBoost Compatibility
    df.columns = df.columns.str.strip()
    df = clean_column_names(df)

    # Drop non-predictive identifier columns if present
    drop_cols = [col for col in ["UDI", "Product_ID"] if col in df.columns]
    df = df.drop(columns=drop_cols)

    # Rename Target Columns for standard naming
    df = df.rename(
        columns={"Machine_failure": "Target", "Type": "Product_Type"}
    )

    # Drop individual failure mode sub-targets to prevent data leakage
    failure_subtypes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    subtypes_to_drop = [c for c in failure_subtypes if c in df.columns]
    df = df.drop(columns=subtypes_to_drop)

    # 3. Domain Physics Feature Engineering
    # Power = Torque (Nm) * Rotational speed (rad/s)
    if "Rotational_speed_rpm" in df.columns and "Torque_Nm" in df.columns:
        df["Power_kW"] = (
            df["Rotational_speed_rpm"]
            * (2 * np.pi / 60)
            * df["Torque_Nm"]
            / 1000
        )

    # Temperature difference (Process Temp - Air Temp)
    if (
        "Process_temperature_K" in df.columns
        and "Air_temperature_K" in df.columns
    ):
        df["Temp_Diff"] = (
            df["Process_temperature_K"] - df["Air_temperature_K"]
        )

    # Tool Wear Strain = Tool wear * Torque
    if "Tool_wear_min" in df.columns and "Torque_Nm" in df.columns:
        df["Tool_Wear_Strain"] = df["Tool_wear_min"] * df["Torque_Nm"]

    # Temp Speed Ratio = Process Temp / Rotational Speed
    if (
        "Process_temperature_K" in df.columns
        and "Rotational_speed_rpm" in df.columns
    ):
        df["Temp_Speed_Ratio"] = (
            df["Process_temperature_K"] / df["Rotational_speed_rpm"]
        )

    # 4. Encode Categorical Features
    os.makedirs("models", exist_ok=True)
    encoder_path = "models/type_encoder.pkl"

    if "Product_Type" in df.columns:
        encoder = LabelEncoder()
        df["Product_Type"] = encoder.fit_transform(df["Product_Type"])
        joblib.dump(encoder, encoder_path)

    # Separate Features and Target
    X = df.drop(columns=["Target"])
    y = df["Target"]

    # 5. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Feature Selection (SelectKBest)
    k = min(k_features, X_train.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    selected_feature_names = X_train.columns[
        selector.get_support()
    ].tolist()
    joblib.dump(selector, "models/feature_selector.pkl")

    # Convert back to DataFrames with clean column names
    X_train = pd.DataFrame(
        X_train_selected, columns=selected_feature_names, index=X_train.index
    )
    X_test = pd.DataFrame(
        X_test_selected, columns=selected_feature_names, index=X_test.index
    )

    # 7. SMOTE Oversampling
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    # 8. Feature Scaling (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "models/scaler.pkl")

    X_train_scaled_df = pd.DataFrame(
        X_train_scaled, columns=selected_feature_names
    )
    X_test_scaled_df = pd.DataFrame(
        X_test_scaled, columns=selected_feature_names
    )

    # 9. Save Processed Datasets to data/processed/
    if save_processed:
        os.makedirs("data/processed", exist_ok=True)
        X_train.to_csv("data/processed/X_train.csv", index=False)
        X_test.to_csv("data/processed/X_test.csv", index=False)
        pd.Series(y_train, name="Target").to_csv(
            "data/processed/y_train.csv", index=False
        )
        pd.Series(y_test, name="Target").to_csv(
            "data/processed/y_test.csv", index=False
        )
        X_train_scaled_df.to_csv(
            "data/processed/X_train_scaled.csv", index=False
        )
        X_test_scaled_df.to_csv(
            "data/processed/X_test_scaled.csv", index=False
        )
        print("Successfully saved processed datasets to data/processed/")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        scaler,
        selected_feature_names,
    )


if __name__ == "__main__":
    preprocess_data(file_path="data/raw/ai4i2020.csv", use_smote=True, k_features=8)