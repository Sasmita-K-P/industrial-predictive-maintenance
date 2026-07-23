import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(model, X_test, y_test, model_name: str, reports_dir: str = "reports"):
    """
    Evaluates a model, prints detailed classification metrics, 
    and saves a Confusion Matrix plot to reports/.
    """
    os.makedirs(reports_dir, exist_ok=True)
    
    y_pred = model.predict(X_test)
    
    # Calculate probabilities for ROC-AUC if supported
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_prob)
    else:
        auc_score = 0.0

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n==================================================")
    print(f" Performance Metrics: {model_name}")
    print(f"==================================================")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    if auc_score > 0:
        print(f"ROC-AUC   : {auc_score:.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save Confusion Matrix Visual
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Normal", "Failure"],
                yticklabels=["Normal", "Failure"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    
    clean_name = model_name.lower().replace(" ", "_")
    plt.savefig(os.path.join(reports_dir, f"cm_{clean_name}.png"))
    plt.close()

    return {
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc_score
    }