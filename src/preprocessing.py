import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)


def load_data(file_path):

    try:

        df = pd.read_csv(file_path)

        logger.info(
            f"Loaded {file_path}"
        )

        return df

    except Exception as e:

        logger.error(str(e))
        raise


def remove_duplicates(df):

    try:

        before = len(df)

        df = df.drop_duplicates()

        after = len(df)

        logger.info(
            f"Removed {before-after} duplicates"
        )

        return df

    except Exception as e:

        logger.error(str(e))
        raise


def handle_missing_values(df):

    try:

        numerical_cols = df.select_dtypes(
            include=["number"]
        ).columns

        categorical_cols = df.select_dtypes(
            include=["object"]
        ).columns

        for col in numerical_cols:

            df[col] = df[col].fillna(
                df[col].median()
            )

        for col in categorical_cols:

            df[col] = df[col].fillna(
                df[col].mode()[0]
            )

        logger.info(
            "Missing values handled"
        )

        return df

    except Exception as e:

        logger.error(str(e))
        raise


def correct_data_types(df):

    try:

        if "signup_time" in df.columns:

            df["signup_time"] = pd.to_datetime(
                df["signup_time"]
            )

        if "purchase_time" in df.columns:

            df["purchase_time"] = pd.to_datetime(
                df["purchase_time"]
            )

        if "age" in df.columns:

            df["age"] = pd.to_numeric(
                df["age"],
                errors="coerce"
            )

        if "purchase_value" in df.columns:

            df["purchase_value"] = pd.to_numeric(
                df["purchase_value"],
                errors="coerce"
            )

        logger.info(
            "Data types corrected"
        )

        return df

    except Exception as e:

        logger.error(str(e))
        raise


def data_cleaning_pipeline(df):

    df = remove_duplicates(df)

    df = correct_data_types(df)

    df = handle_missing_values(df)

    return df

def encode_features(df, categorical_cols):

    """
    One-Hot Encode categorical columns
    """

    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=True
    )

    return df


def scale_features(df, numerical_cols):

    """
    Scale numerical columns
    """

    scaler = StandardScaler()

    df[numerical_cols] = scaler.fit_transform(
        df[numerical_cols]
    )

    return df