import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE


def create_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers physics-based domain features for industrial equipment.
    """
    df = df.copy()

    # 1. Temperature Delta (Delta T = Process Temp - Air Temp) -> Key for HDF
    df["Temp_Difference"] = df["Process_temperature_K"] - df["Air_temperature_K"]

    # 2. Mechanical Power (kW) = (2 * pi * Speed * Torque) / 60000 -> Key for PWF
    df["Power_kW"] = (2 * np.pi * df["Rotational_speed_rpm"] * df["Torque_Nm"]) / 60000.0

    # 3. Tool Wear Strain = Tool Wear * Torque -> Key for OSF
    df["Tool_Wear_Strain"] = df["Tool_wear_min"] * df["Torque_Nm"]

    # 4. Temperature-Speed Ratio -> Thermal-mechanical stress indicator
    df["Temp_Speed_Ratio"] = df["Process_temperature_K"] / (df["Rotational_speed_rpm"] + 1e-5)

    return df


def preprocess_data(path: str, use_smote: bool = True, k_features: int = 8, models_dir: str = "models"):
    """
    Full data preprocessing and feature engineering pipeline.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")

    # Drop non-predictive ID columns
    df.drop(columns=["UDI", "Product ID"], errors="ignore", inplace=True)

    # Standardize column names
    df.columns = [
        col.replace("[", "")
        .replace("]", "")
        .replace("<", "")
        .replace(">", "")
        .replace(" ", "_")
        for col in df.columns
    ]

    print(df.columns.tolist())

    # Encode Categorical Equipment Type ('L', 'M', 'H')
    encoder = LabelEncoder()
    df["Type"] = encoder.fit_transform(df["Type"])
    joblib.dump(encoder, os.path.join(models_dir, "type_encoder.pkl"))

    # Apply Domain-Specific Feature Engineering
    df = create_domain_features(df)

    # Isolate Target and Feature Matrix
    drop_targets = ["Machine_failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
    X = df.drop(columns=[col for col in drop_targets if col in df.columns])
    y = df["Machine_failure"]

    # Feature Selection using Mutual Information
    selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
    X_selected = selector.fit_transform(X, y)
    
    selected_features = X.columns[selector.get_support()]

    print("\nSelected Features:")
    print(selected_features.tolist())

    # Save Feature Selector & Selected Feature names for Inference pipeline
    joblib.dump(selector, os.path.join(models_dir, "feature_selector.pkl"))
    joblib.dump(selected_features.tolist(), os.path.join(models_dir, "selected_features.pkl"))

    # Convert back to DataFrame
    X = pd.DataFrame(X_selected, columns=selected_features)

    # Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nBefore SMOTE:")
    print(y_train.value_counts())

    # SMOTE Oversampling
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    print("\nAfter SMOTE:")
    print(y_train.value_counts())

    print("\n========== Failure Type Distribution ==========\n")
    failure_types = {
        "Tool Wear Failure (TWF)": df["TWF"].sum() if "TWF" in df else 0,
        "Heat Dissipation Failure (HDF)": df["HDF"].sum() if "HDF" in df else 0,
        "Power Failure (PWF)": df["PWF"].sum() if "PWF" in df else 0,
        "Overstrain Failure (OSF)": df["OSF"].sum() if "OSF" in df else 0,
        "Random Failure (RNF)": df["RNF"].sum() if "RNF" in df else 0
    }

    for failure, count in failure_types.items():
        print(f"{failure:<35} : {count}")

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        scaler,
        selected_features
    )