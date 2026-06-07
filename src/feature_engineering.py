import pandas as pd
import logging

logger = logging.getLogger(__name__)

def create_time_features(df):

    try:

        df["signup_time"] = pd.to_datetime(
            df["signup_time"]
        )

        df["purchase_time"] = pd.to_datetime(
            df["purchase_time"]
        )

        df["hour_of_day"] = (
            df["purchase_time"].dt.hour
        )

        df["day_of_week"] = (
            df["purchase_time"].dt.dayofweek
        )

        df["time_since_signup"] = (
            df["purchase_time"]
            -
            df["signup_time"]
        ).dt.total_seconds() / 3600

        logger.info(
            "Time features created successfully"
        )

        return df

        return df

    except Exception as e:

        logger.error(
            f"Error creating time features: {e}"
        )

        raise


def transaction_frequency(df):

    try:

        df["transaction_count"] = (
            df.groupby("user_id")
            ["user_id"]
            .transform("count")
        )

        return df

    except Exception as e:

        logger.error(
            f"Error calculating transaction frequency: {e}"
        )

        raise


def transaction_velocity(df):

    try:

        df["purchase_date"] = (
            df["purchase_time"].dt.date
        )

        df["daily_transactions"] = (
            df.groupby(
                ["user_id", "purchase_date"]
            )["user_id"]
            .transform("count")
        )

        return df

    except Exception as e:

        logger.error(
            f"Error calculating transaction velocity: {e}"
        )

        raise
