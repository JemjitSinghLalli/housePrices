from typing import Callable

import pandas as pd


def select_features_using_lgb_importance(
    train_df: pd.DataFrame,
    features: pd.Series,
    target: str,
    model: Callable,
    min_importance: int,
) -> pd.Series:
    """
    This function will select features from the series of possible features based on their defined by
    lgb in built importance analysis of the features. Uses metric "splits" for importance score.
    Args:
        train_df: The training data
        features: The possible features to select from
        target: The target variable in the datasets
        model: The lgb model to fit, with the keyword args for hyperparameters as optional
        min_importance: Defines the minimum required importance value

    Returns: The Series of features selected
    """
    print(
        f"Length of features passed to `select_features_using_lgb_importance()` {len(features)}"
    )
    model = model.fit(train_df[features], train_df[target])
    importance_df = pd.DataFrame(
        {"feature": features, "importance": model.feature_importances_}
    )

    selected_features_df = importance_df[importance_df["importance"] >= min_importance]
    print(
        f"Length of features selected by `select_features_using_lgb_importance()` {len(selected_features_df)}"
    )
    return selected_features_df["feature"]
