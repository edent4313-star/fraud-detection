import pandas as pd

from src.data_preprocessing import (
    remove_duplicates
)


def test_remove_duplicates():

    df = pd.DataFrame({
        "A": [1,1,2]
    })

    result = remove_duplicates(df)

    assert len(result) == 2