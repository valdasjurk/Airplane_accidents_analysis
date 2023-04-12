# Source: https://www.kaggle.com/datasets/khsamaha/aviation-accident-database-synopses

import kaggle
import pandas as pd

# import numpy as np

DIRECTORY = "data/raw/AviationData.csv"


def download_dataset():
    kaggle.api.dataset_download_files(
        dataset="khsamaha/aviation-accident-database-synopses",
        path="data",
        unzip=True,
        quiet=False,
        force=False,
    )


def read_dataset():
    df = pd.read_csv(DIRECTORY, encoding="cp1252")
    df = df.drop_duplicates()
    return df


def column_name_replacement(df, what_to_replace: str, replacement: str):
    """Replaces symbols, letters in all column names with a given string"""
    df.columns = df.columns.str.replace(what_to_replace, replacement)
    return df


def separate_city_and_state(df):
    """Separates city and state"""
    df[["City", "State"]] = df["Location"].str.split(",", n=1, expand=True)
    return df


def get_year_and_month_from_date(df):
    """Separates year and month from accident date"""
    df["Event_Date"] = pd.to_datetime(df["Event_Date"])
    df["Event_year"] = df["Event_Date"].dt.year
    df["Event_month"] = df["Event_Date"].dt.month
    return df


def get_accident_amount_by_period(df, column_name: str, start_year: int, end_year: int):
    df_by_condition = df[
        (df[column_name] >= start_year) & (df[column_name] <= end_year)
    ]
    amount_of_accidents = len(df_by_condition)
    return amount_of_accidents


def cut_unwanted_parts_from_column_values(df):
    df["Injury_Severity"] = df["Injury_Severity"].str.replace("\W", "", regex=True)
    df["Injury_Severity"] = df["Injury_Severity"].str.replace("\d+", "", regex=True)


def group_by_injury_severity(df):
    grouped_by_severity = df.groupby("Injury_Severity")["Total_Fatal_Injuries"].agg(
        ["min", "max", "sum"]
    )
    return grouped_by_severity


if __name__ == "__main__":
    df = read_dataset()
    df = column_name_replacement(df, ".", "_")
    separate_city_and_state(df)
    df = get_year_and_month_from_date(df)
    print(get_accident_amount_by_period(df, "Event_year", 2020, 2023))
    print(df.head())
    # df = cut_unwanted_parts_from_column_values(df)
    print(group_by_injury_severity(df))
