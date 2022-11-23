
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from utils.features.feature_engineering import engineer_features
from utils.features.feature_selection import select_features_using_lgb_importance
from utils.load.data_importing import import_csv_data
from utils.preprocessing.generic_preprocessing import clean_data, remove_all_outliers, convert_object_to_categorical

features = [
    "lotsize_sq_ft",
    "bedrooms",
    "bathrooms",
    "driveway",
]

fixed_params = {
    "boosting_type": "dart",
    "objective": "regression_l1",
    "metrics": "mae",
    "n_jobs": -1,
    "verbose": -1,
}

hyperparameter_search_space = {
    "num_iterations": (110, 330, 550, 770),
    "learning_rate": (0.1, 0.2),
}

data = import_csv_data("data/house_pricing.csv")
data = clean_data(data_frame=data)
columns_for_outlier_detection = list(["price", "lotsize_sq_ft"])
data = remove_all_outliers(
    data_frame=data,
    columns_for_outlier_detection=columns_for_outlier_detection,
    method="interquartile",
)
data = convert_object_to_categorical(data)
target = "price"
data, features = engineer_features(data_frame=data, target=target)

train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

features = select_features_using_lgb_importance(
    train_df=train_set,
    features=features,
    target=target,
    model=lgb.LGBMRegressor(**fixed_params),
    min_importance=0.5,
)
train_set.head()
