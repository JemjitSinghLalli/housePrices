import lightgbm as lgb
import pandas as pd
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split

from utils.evaluation.model_evaluation import evaluate_regression_model
from utils.features.feature_engineering import engineer_features
from utils.features.feature_selection import select_features_using_lgb_importance
from utils.load.data_importing import import_csv_data
from utils.preprocessing.generic_preprocessing import (
    clean_data,
    remove_all_outliers,
    convert_object_to_categorical,
)


fixed_params = {
    "boosting_type": "gbdt",
    "objective": "regression_l1",
    "metrics": "mae",
    "n_jobs": -1,
    "verbose": -1,
}

hyperparameter_search_space = {
    "num_iterations": (110, 330, 550, 770),
    "learning_rate": (0.1, 0.2),
}


def runner(data_frame: pd.DataFrame, target: str):

    data_frame = clean_data(data_frame=data_frame)
    columns_for_outlier_detection = list(["price", "lotsize_sq_ft"])
    data_frame = remove_all_outliers(
        data_frame=data_frame,
        columns_for_outlier_detection=columns_for_outlier_detection,
        method="interquartile",
    ).drop(columns="Outlier")
    data_frame = convert_object_to_categorical(data_frame)
    data_frame, features = engineer_features(data_frame=data_frame, target=target)

    train_set, test_set = train_test_split(data_frame, test_size=0.2, random_state=42)

    features = select_features_using_lgb_importance(
        train_df=train_set,
        features=features,
        target=target,
        model=lgb.LGBMRegressor(**fixed_params),
        min_importance=0.5,
    )

    model = lgb.LGBMRegressor(**fixed_params)
    model.fit(train_set[features], train_set[target])
    filename = f'model/model_{datetime.today().strftime("%Y-%m-%d")}.pkl'
    pickle.dump(model, open(filename, "wb"))

    return evaluate_regression_model(
        train_df=train_set,
        validate_df=test_set,
        model=lgb.LGBMRegressor(**fixed_params),
        features=features,
        target=target,
    )


if __name__ == "__main__":
    housing_data = import_csv_data("data/house_pricing.csv")
    output = runner(data_frame=housing_data, target="price")
    print(output)
