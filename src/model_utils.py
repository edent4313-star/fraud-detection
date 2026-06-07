import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    print(
        f"Accuracy : {accuracy_score(y_test,y_pred):.4f}"
    )

    print(
        f"Precision: {precision_score(y_test,y_pred):.4f}"
    )

    print(
        f"Recall   : {recall_score(y_test,y_pred):.4f}"
    )

    print(
        f"F1 Score : {f1_score(y_test,y_pred):.4f}"
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    ConfusionMatrixDisplay(
        confusion_matrix=cm
    ).plot(cmap="Blues")

    plt.show()