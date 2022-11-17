import numpy as np
import pandas as pd
import pytest

from utils.preprocessing.generic_preprocessing import clean_data


def test_clean_data():
    unclean_df = pd.DataFrame(
        {
            "nully_numbers": [1.0, 1.1, np.nan, 1.3, 1.4, 1.5, np.nan, 1.7, 1.8, 1.9,],
            "numbers": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
            "no_variance_numbers": [1.3] * 10,
            "strings": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            "no_variance_strings": ["hello"] * 10,
            "nully_booleans": [
                False,
                True,
                None,
                False,
                False,
                False,
                None,
                True,
                True,
                False,
            ],
            "booleans": [True, False] * 5,
            "no_variance_booleans": [True] * 10,
        }
    )

    clean_df = clean_data(unclean_df)
    assert isinstance(
        clean_df, pd.DataFrame
    ), "clean_data() is not returning a dataframe in the first return"
    assert (
        ~clean_df.isna().any().any()
    ), "There are still nulls in the data after cleaning with clean_data()"
