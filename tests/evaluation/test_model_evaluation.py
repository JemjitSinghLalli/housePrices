import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.evaluation.model_evaluation import evaluate_regression_model


def get_test_dataframe():
    np.random.seed(14)
    target_series = np.random.uniform(size=5_000)
    good_feature = [x + np.random.uniform(high=0.25) for x in target_series]
    good_feature1 = [x + np.random.uniform(high=0.3) for x in target_series]
    good_feature2 = [x + np.random.uniform(high=0.35) for x in target_series]

    test_data_df = pd.DataFrame(
        {
            "target": target_series,
            "good_feature": good_feature,
            "good_feature1": good_feature1,
            "good_feature2": good_feature2,
        }
    )
    return test_data_df


def test_evaluate_regression_model():
    test_data_df = get_test_dataframe()
    target = "target"
    features = pd.Series(test_data_df.columns[test_data_df.columns != target])
    train_set, test_set = train_test_split(test_data_df, test_size=0.2, random_state=42)
    test_baseline_error, test_mae, test_r2 = evaluate_regression_model(
        train_df=train_set,
        validate_df=test_set,
        target=target,
        features=features,
        model=lgb.LGBMRegressor(),
    )
    assert all(
        [
            type(output) == float or np.float64
            for output in [test_baseline_error, test_mae, test_r2]
        ]
    ), "the outputs of evaluate_regression_model() are not returning floats"
    assert (
        test_mae < test_baseline_error
    ), "mean absolute error is not coming back from evaluate_regression_model() as low as would be expected"
    assert (
        test_r2 > 0.75
    ), "r-squared score is not coming back from evaluate_regression_model() as high as would be expected"
