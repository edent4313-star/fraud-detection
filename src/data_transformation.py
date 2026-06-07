import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def scale_fraud_data(df):

    """
    Scale numerical features in Fraud_Data.csv
    """

    try:

        numerical_features = [
            "purchase_value",
            "age",
            "time_since_signup",
            "transaction_count",
            "daily_transactions"
        ]

        scaler = StandardScaler()

        df[numerical_features] = scaler.fit_transform(
            df[numerical_features]
        )

        logger.info(
            "Fraud dataset numerical features scaled successfully."
        )

        return df

    except Exception as e:

        logger.error(
            f"Scaling fraud dataset failed: {e}"
        )

        raise


def encode_fraud_data(df):

    """
    One-hot encode Fraud_Data categorical columns
    """

    try:

        categorical_features = [
            "source",
            "browser",
            "sex",
            "country"
        ]

        df = pd.get_dummies(
            df,
            columns=categorical_features,
            drop_first=True
        )

        logger.info(
            "Fraud dataset categorical features encoded successfully."
        )

        return df

    except Exception as e:

        logger.error(
            f"Encoding fraud dataset failed: {e}"
        )

        raise


def scale_creditcard_data(df):

    """
    Scale Time and Amount columns
    """

    try:

        scaler = StandardScaler()

        df[["Time", "Amount"]] = (
            scaler.fit_transform(
                df[["Time", "Amount"]]
            )
        )

        logger.info(
            "Credit card dataset scaled successfully."
        )

        return df

    except Exception as e:

        logger.error(
            f"Scaling credit card dataset failed: {e}"
        )

        raise