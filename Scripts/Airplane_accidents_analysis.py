# Source: https://www.kaggle.com/datasets/khsamaha/aviation-accident-database-synopses

import kaggle
import pandas as pd

import pandera as pa

from pandera.typing import DataFrame, Series

RAW_DATA_DIRECTORY = "data/raw/AviationData.csv"


class InputSchema(pa.SchemaModel):
    Event_Id: Series[str] = pa.Field(coerce=True, nullable=True)
    Investigation_Type: Series[str] = pa.Field(coerce=True, nullable=True)
    Accident_Number: Series[str] = pa.Field(coerce=True, nullable=True)
    Event_Date: Series[pd.DatetimeTZDtype] = pa.Field(
        dtype_kwargs={"unit": "ns", "tz": "EST"}, coerce=True, nullable=True
    )
    Location: Series[str] = pa.Field(coerce=True, nullable=True)
    Country: Series[str] = pa.Field(coerce=True, nullable=True)
    Latitude: Series[str] = pa.Field(coerce=True, nullable=True)
    Longitude: Series[str] = pa.Field(coerce=True, nullable=True)
    Airport_Code: Series[str] = pa.Field(coerce=True, nullable=True)
    Airport_Name: Series[str] = pa.Field(coerce=True, nullable=True)
    Injury_Severity: Series[str] = pa.Field(coerce=True, nullable=True)
    Aircraft_damage: Series[str] = pa.Field(coerce=True, nullable=True)
    Aircraft_Category: Series[str] = pa.Field(coerce=True, nullable=True)
    Registration_Number: Series[str] = pa.Field(coerce=True, nullable=True)
    Make: Series[str] = pa.Field(coerce=True, nullable=True)
    Model: Series[str] = pa.Field(coerce=True, nullable=True)
    Amateur_Built: Series[str] = pa.Field(coerce=True, nullable=True)
    Number_of_Engines: Series[str] = pa.Field(coerce=True, nullable=True)
    Engine_Type: Series[str] = pa.Field(coerce=True, nullable=True)
    FAR_Description: Series[str] = pa.Field(coerce=True, nullable=True)
    Schedule: Series[str] = pa.Field(coerce=True, nullable=True)
    Purpose_of_flight: Series[str] = pa.Field(coerce=True, nullable=True)
    Air_carrier: Series[str] = pa.Field(coerce=True, nullable=True)
    Total_Fatal_Injuries: Series[float] = pa.Field(coerce=True, nullable=True)
    Total_Serious_Injuries: Series[float] = pa.Field(coerce=True, nullable=True)
    Total_Minor_Injuries: Series[float] = pa.Field(coerce=True, nullable=True)
    Total_Uninjured: Series[float] = pa.Field(coerce=True, nullable=True)
    Weather_Condition: Series[str] = pa.Field(coerce=True, nullable=True)
    Broad_phase_of_flight: Series[str] = pa.Field(coerce=True, nullable=True)
    Report_Status: Series[str] = pa.Field(coerce=True, nullable=True)
    Publication_Date: Series[pd.DatetimeTZDtype] = pa.Field(
        dtype_kwargs={"unit": "ns", "tz": "EST"}, coerce=True, nullable=True
    )


class OutputSchema(InputSchema):
    City: Series[str] = pa.Field(nullable=True)
    State: Series[str] = pa.Field(nullable=True)


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
        low_memory=False,
        na_values=" ",
    )
    return df


def column_name_replacement(df, what_to_replace: str, replacement: str) -> pd.DataFrame:
    """Replaces symbols, letters in all column names with a given string"""
    df.columns = df.columns.str.replace(what_to_replace, replacement)
    return df


@pa.check_types
def separate_city_and_state(df: DataFrame[InputSchema]) -> DataFrame[OutputSchema]:
    """Separates city and state"""
    df_city_state = df["Location"].str.split(",", n=1, expand=True)
    return df.assign(City=df_city_state[0], State=df_city_state[1])


def create_year_and_month_column_from_date(df) -> pd.DataFrame:
    """Separates year and month from accident date"""
    # df["Event_Date"] = pd.to_datetime(df["Event_Date"])
    df = df.assign(Event_year=df["Event_Date"].dt.year)
    df = df.assign(Event_month=df["Event_Date"].dt.month)
    return df


def filter_by_period(df, column_name: str, start_year: int, end_year: int):
    return df[(df[column_name] >= start_year) & (df[column_name] <= end_year)]


def get_accident_amount_by_period(df, column_name: str, start_year: int, end_year: int):
    """Returns accident amount by a given period"""
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


def timedelta_between_accident_and_publication(df):
    """Calculates difference in days between publication date and event date"""
    timedelta = (df["Publication_Date"] - df["Event_Date"]).dt.days
    print(timedelta)
    return df.assign(Timedelta=timedelta)


if __name__ == "__main__":
    df = read_dataset()
    df = column_name_replacement(df, ".", "_")
    df_mod = separate_city_and_state(df)
    df_mod = create_year_and_month_column_from_date(df_mod)
    df_mod = remove_symbols_and_digits_from_column(df_mod, "Injury_Severity")
    print(get_accident_amount_by_period(df_mod, "Event_year", 2020, 2023))
    print(get_min_max_sum_death_injuries_by_injury_groups(df_mod))
    timedelta_between_accident_and_publication(df_mod)
    # print(df.head())

    # df = read_dataset()
    # (
    #     df.pipe(column_name_replacement, ".", "_")
    #     .pipe(separate_city_and_state)
    #     .pipe(create_year_and_month_column_from_date)
    #     .pipe(remove_symbols_and_digits_from_column, "Injury_Severity")
    #     .pipe(get_min_max_sum_death_injuries_by_injury_groups)
    # .pipe(get_accident_amount_by_period("Event_year", 2020, 2023))
    # )


""" Functions for final data representation"""
# create_visualization = visualize(df)
# final_report_df = create_report(df)
# df.to_excel("sdfsdf.xlsx")
