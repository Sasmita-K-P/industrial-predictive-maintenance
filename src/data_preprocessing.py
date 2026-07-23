import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

from imblearn.over_sampling import SMOTE

def preprocess_data(path):

    df = pd.read_csv(path)

    print("Shape:", df.shape)

    df.drop(["UDI", "Product ID"], axis=1, inplace=True)

    df.columns = [
        col.replace("[", "")
        .replace("]", "")
        .replace("<", "")
        .replace(">", "")
        .replace(" ", "_")
        for col in df.columns
    ]

    print(df.columns.tolist())

    encoder = LabelEncoder()
    df["Type"] = encoder.fit_transform(df["Type"])

    X = df.drop(
        [
            "Machine_failure",
            "TWF",
            "HDF",
            "PWF",
            "OSF",
            "RNF"
        ],
        axis=1
    )

    y = df["Machine_failure"]

    # Feature Selection
    selector = SelectKBest(
        score_func=mutual_info_classif,
        k=5
    )

    X_selected = selector.fit_transform(X, y)

    selected_features = X.columns[
        selector.get_support()
    ]

    print("\nSelected Features:")
    print(selected_features)

    # Convert back to DataFrame
    X = pd.DataFrame(
        X_selected,
        columns=selected_features
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nBefore SMOTE:")
    print(y_train.value_counts())

    smote = SMOTE(
        random_state=42
    )

    X_train, y_train = smote.fit_resample(
        X_train,
        y_train
    )

    print("\nAfter SMOTE:")
    print(y_train.value_counts())
    print("\n========== Failure Type Distribution ==========\n")

    failure_types = {
        "Tool Wear Failure (TWF)": df["TWF"].sum(),
        "Heat Dissipation Failure (HDF)": df["HDF"].sum(),
        "Power Failure (PWF)": df["PWF"].sum(),
        "Overstrain Failure (OSF)": df["OSF"].sum(),
        "Random Failure (RNF)": df["RNF"].sum()
    }

    for failure, count in failure_types.items():
        print(f"{failure:<35} : {count}")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        scaler,
        X.columns
    )