import matplotlib.pyplot as plt

from sklearn.metrics import (
    f1_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def evaluate_model(
    model,
    X_test,
    y_test
):

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    f1 = f1_score(
        y_test,
        y_pred
    )

    auc_pr = average_precision_score(
        y_test,
        y_prob
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"AUC-PR   : {auc_pr:.4f}"
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot(
        cmap="Blues"
    )

    plt.show()

    return {

        "f1": f1,
        "auc_pr": auc_pr

    }