from sklearn.model_selection import train_test_split

from utils.features.feature_engineering import engineer_features
from utils.load.data_importing import import_csv_data
from utils.preprocessing.generic_preprocessing import clean_data, remove_all_outliers


data = import_csv_data("data/house_pricing.csv")
data = clean_data(data_frame=data)
columns_for_outlier_detection = list(["price", "lotsize_sq_ft"])
data = remove_all_outliers(
    data_frame=data,
    columns_for_outlier_detection=columns_for_outlier_detection,
    method="interquartile",
)
data = engineer_features(data_frame=data)
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
train_set.head()
