import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def explain_model(model, feature_names, X_test=None, reports_dir: str = "reports"):
    """
    Generates both classic Feature Importance and SHAP (SHapley Additive exPlanations)
    summary plots for the champion model.
    """
    os.makedirs(reports_dir, exist_ok=True)
    model_name = type(model).__name__

    print(f"\n[INFO] Generating Explainability Plots for {model_name}...")

    # =========================================================================
    # 1. Standard Feature Importance Bar Plot (Preserved & Upgraded)
    # =========================================================================
    plt.figure(figsize=(10, 6))

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        # Sort features by importance
        indices = np.argsort(importance)
        sorted_features = [feature_names[i] for i in indices]
        sorted_importance = importance[indices]

        plt.barh(sorted_features, sorted_importance, color="#2b5c8f")
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.title(f"Feature Importance ({model_name})")
        plt.tight_layout()
        
        feat_path = os.path.join(reports_dir, "feature_importance.png")
        plt.savefig(feat_path, dpi=300)
        plt.close()
        print(f"✅ Feature Importance plot saved to: {feat_path}")
    else:
        print("[INFO] Model does not have 'feature_importances_' attribute. Skipping bar plot.")

    # =========================================================================
    # 2. Advanced SHAP (SHapley Additive exPlanations) Analysis
    # =========================================================================
    if X_test is not None:
        try:
            # Ensure X_test is a DataFrame with feature names
            if not isinstance(X_test, pd.DataFrame):
                X_sample = pd.DataFrame(X_test, columns=feature_names)
            else:
                X_sample = X_test.copy()

            # Select appropriate SHAP Explainer based on model family
            if model_name in ["XGBClassifier", "LGBMClassifier", "CatBoostClassifier", "RandomForestClassifier"]:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
            else:
                explainer = shap.Explainer(model, X_sample)
                shap_values = explainer(X_sample).values

            # Handle multi-class / list output format for binary classification
            if isinstance(shap_values, list):
                shap_values = shap_values[1] # Target positive class (Failure = 1)
            elif len(np.shape(shap_values)) == 3:
                shap_values = shap_values[:, :, 1]

            # Generate SHAP Summary Beeswarm Plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.title(f"SHAP Feature Impact Summary ({model_name})", pad=20)
            plt.tight_layout()

            shap_path = os.path.join(reports_dir, "shap_summary.png")
            plt.savefig(shap_path, bbox_inches="tight", dpi=300)
            plt.close()
            print(f"✅ SHAP Summary plot saved to: {shap_path}")

        except Exception as e:
            print(f"⚠️ SHAP explanation skipped due to error: {e}")
    else:
        print("[INFO] X_test not provided. Skipping SHAP generation.")