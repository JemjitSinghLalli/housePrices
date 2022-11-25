import pandas as pd


def engineer_features(data_frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Dataframe holding house prices which engineers features.
    Args:
        data_frame: The dataframe holding columns to feature engineer.
    Returns: the dataframe with engineered features
    """
    data_frame["price_per_sq_ft"] = data_frame["price"] / data_frame["lotsize_sq_ft"]
    data_frame["price_per_bedroom"] = data_frame["price"] / data_frame["bedrooms"]
    data_frame["price_per_bathroom"] = data_frame["price"] / data_frame["bathrooms"]
    data_frame["price_per_stories"] = data_frame["price"] / data_frame["stories"]

    print("Feature engineering is complete.")

    return data_frame, pd.Series(data_frame.drop(columns=target).columns)
