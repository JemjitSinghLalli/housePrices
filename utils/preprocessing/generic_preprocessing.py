import numpy as np
import pandas as pd

from sklearn.neighbors import LocalOutlierFactor


def clean_data(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    This needs to solve for issues in the data that mean it cannot be passed in to a model object, for example if there
    are nulls or infinity values in the data, it also handles columns without variance.
    Args:
        data_frame: The data to clean
    Returns: the data cleaned
    """
    data_frame = data_frame.copy()
    start_rows = data_frame.shape[0]

    data_frame = data_frame.dropna()

    end_rows = data_frame.shape[0]
    print(
        f"{start_rows - end_rows} of the {start_rows} rows were dropped during the clean_data() function."
    )
    print(f"{start_rows - (start_rows - end_rows)} rows remain.")
    return data_frame


def remove_outliers_using_interquartile(
    data_frame: pd.DataFrame, column_name: str
) -> pd.DataFrame:
    """
    This function will remove outliers based on the column name from the provided dataframe using a 'traditional'
    method of interquartile range or boxplot method.

    Args:
        data_frame: The dataframe which holds the column to be analysed for outliers.
        column_name: Column to be analysed for outliers.
    Returns: the dataframe excluding outliers based on the provided column.
    """
    data_frame = data_frame.copy()
    quartile_1 = data_frame[column_name].quantile(0.25)
    quartile_3 = data_frame[column_name].quantile(0.75)
    interquartile_range = quartile_3 - quartile_1
    data_frame["Outlier"] = np.where(
        data_frame[[column_name]] < (quartile_1 - 1.5 * interquartile_range),
        1,
        np.where(
            data_frame[[column_name]] > (quartile_3 + 1.5 * interquartile_range), 1, 0
        ),
    )

    len(data_frame[data_frame["Outlier"] == 1])

    print(
        f'Identified {len(data_frame[data_frame["Outlier"] == 1])} outliers for column {column_name}.'
    )
    return data_frame[data_frame["Outlier"] == 0]


def remove_outliers_using_local_factor(
    data_frame: pd.DataFrame, column_name: str
) -> pd.DataFrame:
    """
    This function will remove outliers based on the column name from the provided dataframe using local deviation of
     the density with respect to its neighbors.

    Args:
        data_frame: The dataframe which holds the column to be analysed for outliers.
        column_name: Column to be analysed for outliers.
    Returns: the dataframe excluding outliers based on the provided column.
    """
    data_frame = data_frame.copy()
    data_frame["Outlier"] = (
        LocalOutlierFactor(n_neighbors=5, novelty=True)
        .fit(data_frame[[column_name]])
        .predict(data_frame[[column_name]])
    )
    print(
        f'Identified {len(data_frame[data_frame["Outlier"] == -1])} outliers for column {column_name}.'
    )
    return data_frame[data_frame["Outlier"] == 1]


def remove_all_outliers(
    data_frame: pd.DataFrame,
    columns_for_outlier_detection: list,
    method: str = ["interquartile", "local_factor"],
) -> pd.DataFrame:
    """
    This function will remove outliers based on the column name from the provided dataframe using a 'traditional'
    method of interquartile range or boxplot method.

    Args:
        data_frame: The dataframe which holds the column to be analysed for outliers.
        columns_for_outlier_detection: Column to be analysed for outliers.
        method: type of outlier methodology to be used
    Returns: the dataframe excluding outliers based on the provided column.
    """
    start_rows = data_frame.shape[0]
    data_frame = data_frame.copy()

    if method not in ["interquartile", "local_factor"]:
        raise Exception("Method for outlier detection is not selected from given list.")

    if len(columns_for_outlier_detection) == 0:
        columns_for_outlier_detection = {
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
            "float64",
        }
        columns_for_outlier_detection = data_frame.select_dtypes(
            include=columns_for_outlier_detection
        ).columns

    if method == "interquartile":
        for col in columns_for_outlier_detection:
            data_frame = remove_outliers_using_interquartile(data_frame, col)
    elif method == "local_factor":
        for col in columns_for_outlier_detection:
            data_frame = remove_outliers_using_local_factor(data_frame, col)

    end_rows = data_frame.shape[0]

    print(
        f"{start_rows - end_rows} of the {start_rows} rows were dropped during the remove_all_outliers() function."
    )
    print(f"{start_rows - (start_rows-end_rows)} rows remain.")

    return data_frame.drop(columns="Outlier")


def convert_object_to_categorical(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    This function will identify features which are object that should be categorical. In order for
    select_features_using_lgb_importance() to work, they need to be categorical.

    Args:
        data_frame: The dataframe which holds all features.
    Returns: the dataframe returned holds categorical features as opposed to object features.
    """
    object_features = data_frame.select_dtypes(["object"]).columns
    data_frame[object_features] = data_frame[object_features].astype("category")

    return data_frame
