import numpy as np
import pandas as pd

from utils.preprocessing.generic_preprocessing import (
    clean_data,
    remove_outliers_using_interquartile,
    remove_outliers_using_local_factor,
    remove_all_outliers,
)


def test_clean_data():
    unclean_df = pd.DataFrame(
        {
            "nully_numbers": [1.0, 1.1, np.nan, 1.3, 1.4, 1.5, np.nan, 1.7, 1.8, 1.9,],
            "numbers": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
            "no_variance_numbers": [1.3] * 10,
            "strings": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            "no_variance_strings": ["hello"] * 10,
            "nully_booleans": [
                False,
                True,
                None,
                False,
                False,
                False,
                None,
                True,
                True,
                False,
            ],
            "booleans": [True, False] * 5,
            "no_variance_booleans": [True] * 10,
        }
    )

    clean_df = clean_data(unclean_df)
    assert isinstance(
        clean_df, pd.DataFrame
    ), "clean_data() is not returning a dataframe in the first return"
    assert (
        ~clean_df.isna().any().any()
    ), "There are still nulls in the data after cleaning with clean_data()"


def test_remove_outliers_using_interquartile():
    test_df = pd.DataFrame(
        [
            ["Ajitesh", 84, 183, "no"],
            ["Shailesh", 79, 186, "yes"],
            ["Seema", 67, 158, "yes"],
            ["Nidhi", 52, 155, "no"],
            ["Ajay", 132, 158, "yes"],
            ["Jimmy", 67, 158, "yes"],
            ["Aaron", 145, 158, "yes"],
            ["Jimmy", 67, 158, "yes"],
            ["Luke", 67, 158, "yes"],
            ["Rob", 7, 158, "yes"],
            ["Dan", 67, 158, "yes"],
        ]
    )
    test_df.columns = ["name", "weight", "height", "smoke_or_not"]
    test_df_removed_outliers = remove_outliers_using_interquartile(test_df, "height")

    assert isinstance(
        test_df_removed_outliers, pd.DataFrame
    ), "returned object is not of expected type, should be a dataframe."
    assert len(test_df_removed_outliers.columns) - len(test_df.columns) == 1, (
        "Outlier column was not appended to the dataframe, please check to see if outlier detection has worked "
        "successfully."
    )
    assert (
        test_df_removed_outliers["Outlier"].unique() == 0
    ), "Dataframe has not removed outliers, some outliers still remain."


def test_remove_outliers_using_local_factor():
    test_df = pd.DataFrame(
        [
            ["Ajitesh", 84, 183, "no"],
            ["Shailesh", 79, 186, "yes"],
            ["Seema", 67, 158, "yes"],
            ["Nidhi", 52, 155, "no"],
            ["Ajay", 132, 158, "yes"],
            ["Jimmy", 67, 158, "yes"],
            ["Aaron", 145, 158, "yes"],
            ["Jimmy", 67, 158, "yes"],
            ["Luke", 67, 158, "yes"],
            ["Rob", 7, 158, "yes"],
            ["Dan", 67, 158, "yes"],
        ]
    )
    test_df.columns = ["name", "weight", "height", "smoke_or_not"]
    test_df_removed_outliers = remove_outliers_using_local_factor(test_df, "height")

    assert isinstance(
        test_df_removed_outliers, pd.DataFrame
    ), "returned object is not of expected type, should be a dataframe."
    assert len(test_df_removed_outliers.columns) - len(test_df.columns) == 1, (
        "Outlier column was not appended to the dataframe, please check to see if outlier detection has worked "
        "successfully."
    )
    assert (
        test_df_removed_outliers["Outlier"].unique() == 1
    ), "Dataframe has not removed outliers, some outliers still remain."


def test_remove_all_outliers():
    test_df = pd.DataFrame(
        [
            ["Ajitesh", 84, 183, "no"],
            ["Shailesh", 79, 186, "yes"],
            ["Seema", 67, 158, "yes"],
            ["Nidhi", 52, 155, "no"],
            ["Ajay", 132, 158, "yes"],
            ["Jimmy", 67, 158, "yes"],
            ["Aaron", 145, 158, "yes"],
            ["Jimmy", 67, 158, "yes"],
            ["Luke", 67, 158, "yes"],
            ["Rob", 7, 158, "yes"],
            ["Dan", 67, 158, "yes"],
        ]
    )
    test_df.columns = ["name", "weight", "height", "smoke_or_not"]
    test_columns_for_outlier_detection = ["weight", "height"]
    method = "interquartile"

    test_df_removed_all_outliers = remove_all_outliers(
        test_df, test_columns_for_outlier_detection, method
    )

    assert isinstance(
        test_df_removed_all_outliers, pd.DataFrame
    ), "returned object is not of expected type, should be a dataframe."
    assert len(test_df_removed_all_outliers.columns) - len(test_df.columns) == 1, (
        "Outlier column was not appended to the dataframe, please check to see if outlier detection has worked "
        "successfully."
    )
    assert (
        test_df_removed_all_outliers["Outlier"].unique() == 1
        or test_df_removed_all_outliers["Outlier"].unique() == 0
    ), "Dataframe has not removed outliers, some outliers still remain."
