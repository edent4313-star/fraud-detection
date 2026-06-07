import pandas as pd
import logging

from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE
)

logger = logging.getLogger(__name__)


def random_oversample(X_train, y_train):

    try:

        ros = RandomOverSampler(
            sampling_strategy=0.5,
            random_state=42
        )

        X, y = ros.fit_resample(
            X_train,
            y_train
        )

        logger.info("Random oversampling completed")

        return X, y

    except Exception as e:

        logger.error(
            f"Oversampling failed: {e}"
        )

        raise


def apply_smote(X_train, y_train):

    try:

        smote = SMOTE(
            sampling_strategy=0.5,
            random_state=42,
            k_neighbors=5
        )

        X, y = smote.fit_resample(
            X_train,
            y_train
        )

        logger.info("SMOTE completed")

        return X, y

    except Exception as e:

        logger.error(
            f"SMOTE failed: {e}"
        )
        raise