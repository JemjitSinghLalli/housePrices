import os

import pandas as pd

from utils.features.feature_engineering import engineer_features
from utils.load.data_importing import import_csv_data

path_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_engineer_features():
    test_data = import_csv_data(f"{path_dir}\\data\\house_pricing.csv")
    test_target = "price"
    test_data_features, features = engineer_features(test_data, test_target)

    assert isinstance(
        test_data_features, pd.DataFrame
    ), "returned dataframe from feature engineering did not return pd.DataFrame"
    assert isinstance(
        features, pd.Series
    ), "returned series from feature engineering did not return pd.Series"
    # TODO: write further tests
