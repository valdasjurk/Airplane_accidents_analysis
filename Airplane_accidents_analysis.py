# Source: https://www.kaggle.com/datasets/khsamaha/aviation-accident-database-synopses

import kaggle
import pandas as pd


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


def reasonable_masks(df):
    df["Event.Date"] = pd.to_datetime(df["Event.Date"])
    year_filtering = df[df["Event.Date"].dt.year > 2020]
    print(year_filtering)
    city_name_mask = df["Location"].str.contains("SEASIDE")
    print(city_name_mask)
    df_by_city_name = df[df["Location"] == "SEASIDE HEIGHTS, NJ"]
    print(df_by_city_name)
    df_outside_usa = df[df["Country"] != "United States"]
    print(df_outside_usa)
    no_of_engines_mask = df.query("Number_of_Engines > 1")
    print(no_of_engines_mask)
    make_mask = df["Make"].isin(["Stinson", "Cessna", "Piper"])
    print(make_mask)


def column_name_replacement(df):
    df.columns = df.columns.str.replace(".", "_")
    return df


def column_to_datetime(df, column_name):
    df[column_name] = pd.to_datetime(df[column_name])
    return df


if __name__ == "__main__":
    # download_dataset()
    df = read_dataset()
    df = column_name_replacement(df)
    df = column_to_datetime(df, "Event_Date")
    print(df["Event_Date"])

    # reasonable_masks(df)
