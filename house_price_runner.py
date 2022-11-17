from utils.load.data_importing import import_csv_data
from utils.preprocessing.generic_preprocessing import clean_data, remove_all_outliers


data = import_csv_data("data\house_pricing.csv")
data = clean_data(data_frame=data)
columns_for_outlier_detection = list(["price", "lotsize_sq_ft"])
data = remove_all_outliers(
    data_frame=data,
    columns_for_outlier_detection=columns_for_outlier_detection,
    method="elpitic_envelope",
)
