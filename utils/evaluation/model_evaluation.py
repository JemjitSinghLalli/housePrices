from typing import Any, Tuple

import pandas as pd
from sklearn.metrics import r2_score


def evaluate_regression_model(
    train_df: pd.DataFrame,
    validate_df: pd.DataFrame,
    target: str,
    features: pd.Series,
    model: Any,
) -> Tuple[float, float, float]:
    """
    This function uses the pre-defined model to predict in the validate set, based on selected features,
    it then returns the baseline error and mean absolute error of those predictions against the validate
    sets true target values.
    Args:
        train_df: The training data for the model
        validate_df: The validation data for the model, aka the holdout set
        features: The features to evaluate performance on
        target: The target in the data
        model: The untrained instance of the model class with sklearn-like methods / attributes

    Returns: Mean absolute error on the validation set
    """
    train_df = train_df.copy()
    validate_df = validate_df.copy()
    fit_model = model.fit(train_df[features], train_df[target])
    y_true = validate_df[target].astype(float)
    y_pred = fit_model.predict(validate_df[features])
    baseline_error = abs(y_true - y_true.mean()).mean()
    mean_absolute_error = abs(y_true - y_pred).mean()
    r2 = r2_score(y_true, y_pred)
    print(f"Target baseline error was evaluated as: {baseline_error}")
    print(
        f"Model was evaluated as having a mean absolute error of: {mean_absolute_error}"
    )
    print(f"Model was evaluated as having an r-squared score of: {r2}")
    return baseline_error, mean_absolute_error, r2
