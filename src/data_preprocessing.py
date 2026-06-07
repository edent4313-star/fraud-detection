import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {file_path}")
        return df

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise

    except Exception as e:
        logger.error(str(e))
        raise


def remove_duplicates(df):
    try:
        before = len(df)

        df = df.drop_duplicates()

        after = len(df)

        logger.info(
            f"Removed {before-after} duplicate rows"
        )

        return df

    except Exception as e:
        logger.error(str(e))
        raise


def handle_missing_values(df):
    try:
        for col in df.select_dtypes(include="number"):
            df[col] = df[col].fillna(df[col].median())

        for col in df.select_dtypes(include="object"):
            df[col] = df[col].fillna(df[col].mode()[0])

        return df

    except Exception as e:
        logger.error(str(e))
        raise