# Source: https://www.kaggle.com/datasets/khsamaha/aviation-accident-database-synopses

import kaggle
import pandas as pd

RAW_DATA_DIRECTORY = "data/raw/AviationData.csv"


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
        RAW_DATA_DIRECTORY,
        parse_dates=["Event.Date", "Publication.Date"],
        encoding="cp1252",
    )
    return df


def column_name_replacement(df, what_to_replace: str, replacement: str) -> pd.DataFrame:
    """Replaces symbols, letters in all column names with a given string"""
    df.columns = df.columns.str.replace(what_to_replace, replacement)
    return df


def separate_city_and_state(df) -> pd.DataFrame:
    """Separates city and state"""
    df[["City", "State"]] = df["Location"].str.split(",", n=1, expand=True)
    return df


def get_year_and_month_from_date(df) -> pd.DataFrame:
    """Separates year and month from accident date"""
    df["Event_Date"] = pd.to_datetime(df["Event_Date"])
    df["Event_year"] = df["Event_Date"].dt.year
    df["Event_month"] = df["Event_Date"].dt.month
    return df


def filter_by_period(df, column_name: str, start_year: int, end_year: int):
    return df[(df[column_name] >= start_year) & (df[column_name] <= end_year)]


def get_accident_amount_by_period(df, column_name: str, start_year: int, end_year: int):
    df_by_condition = df[
        (df[column_name] >= start_year) & (df[column_name] <= end_year)
    ]
    amount_of_accidents = len(df_by_condition)
    return amount_of_accidents


def remove_symbols_and_digits_from_column(df, column_name):
    df[column_name] = df[column_name].str.replace(r"\W", "", regex=True)
    df[column_name] = df[column_name].str.replace(r"\d+", "", regex=True)
    return df


def get_min_max_sum_death_injuries_by_injury_groups(df):
    """Returns Injury severity groups and min, max and sum death accidents"""
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
    df = remove_symbols_and_digits_from_column(df, "Injury_Severity")
    print(df[["Injury_Severity"]])
    print(get_min_max_sum_death_injuries_by_injury_groups(df))


""" Df pipe"""
# df = read_dataset()
# (
#     df.pipe(column_name_replacement, ".", "_")
#     .pipe(separate_city_and_state)
#     .pipe(get_year_and_month_from_date)
# )


""" Functions for final data representation"""
# create_visualization = visualize(df)
# final_report_df = create_report(df)
# df.to_excel("sdfsdf.xlsx")
