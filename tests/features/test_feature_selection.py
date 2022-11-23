import lightgbm as lgb
import pandas as pd

from utils.features.feature_selection import select_features_using_lgb_importance


def test_select_features_using_lgb_importance():
    test_data_df = get_test_dataframe()
    train_df, _ = split_data(test_data_df)
    target = "target"
    features = pd.Series(test_data_df.columns[test_data_df.columns != "target"]).sample(
        frac=1.0
    )
    selected_hyperparams = {"n_estimators": 20}

    selected_features_lgb_test = select_features_using_lgb_importance(
        train_df,
        features,
        target,
        lgb.LGBMRegressor(**selected_hyperparams),
        min_importance=25,
    )

    assert isinstance(
        selected_features_lgb_test, pd.Series
    ), "selected_features_test_series is not returning a series"

    assert all(
        [
            bad_feature not in list(selected_features_lgb_test)
            for bad_feature in ["bad_feature1", "bad_feature2", "bad_feature3"]
        ]
    ), "selected_features_test_series is returning bad features"

