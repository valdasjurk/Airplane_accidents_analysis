# Source: https://www.kaggle.com/datasets/khsamaha/aviation-accident-database-synopses

import kaggle
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import json
import numpy as np
from datetime import timedelta

RAW_DATA_DIRECTORY = "data/raw/AviationData.csv"
INTERIM_DIRECTORY = "data/interim/AviationData_preprocessed.csv"


class InputSchema(pa.SchemaModel):
    Event_Id: Series[str]
    Investigation_Type: Series[str]
    Accident_Number: Series[str]
    Event_Date: Series[pd.DatetimeTZDtype] = pa.Field(
        dtype_kwargs={"unit": "ns", "tz": "EST"}
    )
    Location: Series[str] = pa.Field(nullable=True)
    Country: Series[str] = pa.Field(nullable=True)
    Latitude: Series[str] = pa.Field(nullable=True)
    Longitude: Series[str] = pa.Field(nullable=True)
    Airport_Code: Series[str] = pa.Field(nullable=True)
    Airport_Name: Series[str] = pa.Field(nullable=True)
    Injury_Severity: Series[str] = pa.Field(nullable=True)
    Aircraft_damage: Series[str] = pa.Field(nullable=True)
    Aircraft_Category: Series[str] = pa.Field(nullable=True)
    Registration_Number: Series[str] = pa.Field(nullable=True)
    Make: Series[str] = pa.Field(nullable=True)
    Model: Series[str] = pa.Field(nullable=True)
    Amateur_Built: Series[str] = pa.Field(nullable=True)
    Number_of_Engines: Series[float] = pa.Field(nullable=True)
    Engine_Type: Series[str] = pa.Field(nullable=True)
    FAR_Description: Series[str] = pa.Field(nullable=True)
    Schedule: Series[str] = pa.Field(nullable=True)
    Purpose_of_flight: Series[str] = pa.Field(nullable=True)
    Air_carrier: Series[str] = pa.Field(nullable=True)
    Total_Fatal_Injuries: Series[float] = pa.Field(nullable=True)
    Total_Serious_Injuries: Series[float] = pa.Field(nullable=True)
    Total_Minor_Injuries: Series[float] = pa.Field(nullable=True)
    Total_Uninjured: Series[float] = pa.Field(nullable=True)
    Weather_Condition: Series[str] = pa.Field(nullable=True)
    Broad_phase_of_flight: Series[str] = pa.Field(nullable=True)
    Report_Status: Series[str] = pa.Field(nullable=True)
    Publication_Date: Series[pd.DatetimeTZDtype] = pa.Field(
        dtype_kwargs={"unit": "ns", "tz": "EST"}, nullable=True
    )

    class Config:
        """Input schema config"""

        coerce = True


class OutputSchema(InputSchema):
    pass


def download_dataset() -> None:
    kaggle.api.dataset_download_files(
        dataset="khsamaha/aviation-accident-database-synopses",
        path="data",
        unzip=True,
        quiet=False,
        force=False,
    )


def read_dataset() -> pd.DataFrame:
    df = pd.read_csv(
        RAW_DATA_DIRECTORY,
        parse_dates=["Event.Date", "Publication.Date"],
        encoding="cp1252",
        low_memory=False,
        # na_values=" ",
    )
    return df


def save_to_csv(df: pd.DataFrame) -> None:
    df.to_csv(INTERIM_DIRECTORY, index=False)


@pa.check_types
def column_name_replacement(
    df, what_to_replace: str, replacement: str
) -> DataFrame[InputSchema]:
    """Replaces symbols, letters in all column names with a given string"""
    df.columns = df.columns.str.replace(what_to_replace, replacement)
    return df


def separate_city_and_state(df: pd.DataFrame) -> pd.DataFrame:
    """Separates city and state"""
    df_city_state = df["Location"].str.split(",", n=1, expand=True)
    dfr = df.assign(City=df_city_state[0], State=df_city_state[1])
    return dfr


def create_year_and_month_column_from_date(df: pd.DataFrame) -> pd.DataFrame:
    """Separates year and month from accident date"""
    df = df.assign(Event_year=df["Event_Date"].dt.year)
    df = df.assign(Event_month=df["Event_Date"].dt.month)
    return df


def filter_by_period(
    df, column_name: str, start_year: int, end_year: int
) -> pd.DataFrame:
    return df[(df[column_name] >= start_year) & (df[column_name] <= end_year)]


def get_accident_amount_by_period(
    df, column_name: str, start_year: int, end_year: int
) -> int:
    """Returns accident amount by a given period"""
    df_by_condition = df[
        (df[column_name] >= start_year) & (df[column_name] <= end_year)
    ]
    amount_of_accidents = len(df_by_condition)
    return amount_of_accidents


def remove_symbols_and_digits_from_column(
    df: pd.DataFrame, column_name: str
) -> pd.DataFrame:
    df[column_name] = df[column_name].str.replace(r"\W", "", regex=True)
    df[column_name] = df[column_name].str.replace(r"\d+", "", regex=True)
    return df


def get_min_max_sum_death_injuries_by_injury_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Returns Injury severity groups and min, max and sum death accidents"""
    grouped_by_severity = df.groupby("Injury_Severity")["Total_Fatal_Injuries"].agg(
        ["min", "max", "sum"]
    )
    return grouped_by_severity


def timedelta_between_accident_and_publication(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates difference in days between publication date and event date"""
    timedelta = (df["Publication_Date"] - df["Event_Date"]).dt.days
    return df.assign(Time_between_publication_and_event=timedelta)


def plot_accidents_amount_by_state(df: pd.DataFrame) -> None:
    """plotting events by states count"""
    sns.countplot(
        y="State",
        data=df[df["Country"] == "United States"],
    )
    plt.show()


def plot_time_between_publication_and_event(df: pd.DataFrame) -> None:
    """plotting histogram of time between publication and event"""
    df["Time_between_publication_and_event"].plot.hist(
        bins=12, legend=True, xlim=(0, 6000)
    )
    plt.show()


def preprocese_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df_processed = (
        df.pipe(column_name_replacement, ".", "_")
        .pipe(separate_city_and_state)
        .pipe(create_year_and_month_column_from_date)
        .pipe(remove_symbols_and_digits_from_column, "Injury_Severity")
        .pipe(timedelta_between_accident_and_publication)
    )
    return df_processed


def add_data_from_external_source(df: pd.DataFrame) -> pd.DataFrame:
    """Reads data (temperature by day) from Weatherbit.io regarding accident data"""
    temperatures = []
    df = df.head()
    for index, row in df.iterrows():
        try:
            start_date = row["Event_Date"].date()
            end_date = start_date + timedelta(days=1)

            params = {
                "key": ["b7c8b0bdb6724a3c9d40c8fef07ee335"],
                "start_date": start_date,
                "end_date": end_date,
                "city": row["City"],
            }

            method = "ping"
            api_base = "https://api.weatherbit.io/v2.0/history/daily?"
            api_result = requests.get(api_base + method, params)
            response = api_result.text
            response_dict = json.loads(response)
            for key in response_dict["data"]:
                temperatures.append(key["temp"])
        except ValueError:
            temperatures.append(np.nan)
    return df.assign(Temperatures_accident_day=temperatures)


if __name__ == "__main__":
    df = read_dataset()
    df_processed = preprocese_dataset(df)

    death_injuries_statistics = get_min_max_sum_death_injuries_by_injury_groups(
        df_processed
    )
    accidents_by_period = get_accident_amount_by_period(
        df_processed, "Event_year", 2020, 2023
    )

    add_data_from_external_source(df_processed)


""" Future Functions for final data representation"""
# plot_time_between_publication_and_event(df_processed)
# save_to_csv(df_temp)
# create_visualization = visualize(df)
# final_report_df = create_report(df)
# df.to_excel("sdfsdf.xlsx")
