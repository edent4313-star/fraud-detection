import shap
import matplotlib.pyplot as plt
import pandas as pd


def plot_feature_importance(model, X, top_n=10):

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    ).head(top_n)

    plt.figure(figsize=(10,6))

    plt.barh(
        importance["Feature"],
        importance["Importance"]
    )

    plt.title(
        f"Top {top_n} Feature Importance"
    )

    plt.gca().invert_yaxis()

    plt.show()

    return importance


def create_shap_explainer(model):

    return shap.TreeExplainer(model)


def shap_summary_plot(explainer, X):

    shap_values = explainer.shap_values(X)

    shap.summary_plot(
        shap_values,
        X
    )

    return shap_values


def shap_force_plot(
    explainer,
    shap_values,
    X,
    index
):

    return shap.force_plot(
        explainer.expected_value,
        shap_values[index],
        X.iloc[index],
        matplotlib=True
    )