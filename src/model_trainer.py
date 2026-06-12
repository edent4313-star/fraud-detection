from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    GridSearchCV
)

class FraudModelTrainer:

    def __init__(self, model_type="lr"):

        self.model_type = model_type

        if model_type == "lr":

            self.model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )

        elif model_type == "rf":

            self.model = RandomForestClassifier(
                random_state=42
            )

        elif model_type == "xgb":

            self.model = XGBClassifier(
                random_state=42,
                eval_metric="logloss"
            )

        else:

            raise ValueError("Unsupported model type")

    def cross_validate_model(self, X, y):

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        scores = cross_validate(
            self.model,
            X,
            y,
            cv=cv,
            scoring=["f1", "average_precision"]
        )

        return {

            "mean_f1":
            scores["test_f1"].mean(),

            "std_f1":
            scores["test_f1"].std(),

            "mean_auc_pr":
            scores["test_average_precision"].mean(),

            "std_auc_pr":
            scores["test_average_precision"].std()

        }

    def train(self, X_train, y_train):

        self.model.fit(
            X_train,
            y_train
        )

        return self.model

    def tune_model(
        self,
        X_train,
        y_train
    ):

        if self.model_type == "rf":

            param_grid = {

                "n_estimators":
                [100, 200],

                "max_depth":
                [5, 10, 15]

            }

        elif self.model_type == "xgb":

            param_grid = {

                "n_estimators":
                [100, 200],

                "max_depth":
                [3, 5, 7]

            }

        else:

            return self.model

        grid = GridSearchCV(
            self.model,
            param_grid,
            cv=5,
            scoring="f1",
            n_jobs=-1
        )

        grid.fit(
            X_train,
            y_train
        )

        self.model = grid.best_estimator_

        print(
            "Best Parameters:",
            grid.best_params_
        )

        return self.model