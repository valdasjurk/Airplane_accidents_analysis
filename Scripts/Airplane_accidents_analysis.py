# Source: https://www.kaggle.com/datasets/khsamaha/aviation-accident-database-synopses

import kaggle
import pandas as pd

# import numpy as np

DIRECTORY = "data/raw/AviationData.csv"


def download_dataset() -> None:
    kaggle.api.dataset_download_files(
        dataset="khsamaha/aviation-accident-database-synopses",
        path="data",
        unzip=True,
        quiet=False,
        force=False,
    )


def read_dataset():
    df = pd.read_csv(
        DIRECTORY,
        index_col=0,
        encoding="cp1252",
        dtype={"name": "string", "Event_Date": "datetime64"},
    )
    return df


def column_name_replacement(df, what_to_replace: str, replacement: str) -> pd.DataFrame:
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


def test_get_year_and_month_from_date():
    df = pd.DataFrame({"Event_Date": ["2022-01-01", "2023-02-02"]})
    df_expected = pd.DataFrame(
        {
            "Event_Date": [pd.Timestamp(2022, 1, 1), pd.Timestamp(2023, 2, 2)],
            "Event_year": [2022, 2023],
            "Event_month": [1, 2],
        }
    )
    assert pd.testing.assert_frame_equal(get_year_and_month_from_date(df), df_expected)


def filter_by_period(df, column_name: str, start_year: int, end_year: int):
    return df[(df[column_name] >= start_year) & (df[column_name] <= end_year)]


def get_accident_amount_by_period(df, column_name: str, start_year: int, end_year: int):
    df_by_condition = df[
        (df[column_name] >= start_year) & (df[column_name] <= end_year)
    ]
    amount_of_accidents = len(df_by_condition)
    return amount_of_accidents


def cut_unwanted_parts_from_Injury_Severity(df):
    # def filter_out_non_digits
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
    df = separate_city_and_state(df)
    df = get_year_and_month_from_date(df)
    print(get_accident_amount_by_period(df, "Event_year", 2020, 2023))
    print(df.head())
    # df = cut_unwanted_parts_from_column_values(df)
    print(group_by_injury_severity(df))

    df = read_dataset()
    (
        df.pipe(column_name_replacement, ".", "_")
        .pipe(separate_city_and_state)
        .pipe(get_year_and_month_from_date)
    )

    create_visualization = visualize(df)
    final_report_df = create_report(df)
    df.to_excel("sdfsdf.xlsx")
