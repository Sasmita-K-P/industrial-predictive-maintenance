from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name
):

    pred = model.predict(X_test)

    print("\n")
    print("=" * 50)
    print(model_name)

    print(
        "Accuracy:",
        accuracy_score(y_test, pred)
    )

    print(
        "Precision:",
        precision_score(y_test, pred)
    )

    print(
        "Recall:",
        recall_score(y_test, pred)
    )

    print(
        "F1:",
        f1_score(y_test, pred)
    )

    print(
        classification_report(
            y_test,
            pred
        )
    )