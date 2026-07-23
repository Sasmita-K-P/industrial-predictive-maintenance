import matplotlib.pyplot as plt


def explain_model(
    model,
    feature_names
):

    importance = model.feature_importances_

    plt.figure(figsize=(10,6))

    plt.barh(
        feature_names,
        importance
    )

    plt.xlabel("Importance Score")

    plt.ylabel("Features")

    plt.title(
        "XGBoost Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        "reports/feature_importance.png"
    )

    plt.close()

    print(
        "Feature Importance Plot Saved"
    )