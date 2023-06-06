# Source: https://www.kaggle.com/datasets/khsamaha/aviation-accident-database-synopses

import logging
from datetime import timedelta

import kaggle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandera as pa
import requests
import seaborn as sns
from pandera.typing import DataFrame, Series
import config
import zipfile
import os


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


def logger_df(func):
    def inner(*args, **kwargs):
        logging.info(f"Started executing function: {func.__name__}")
        output = func(*args, **kwargs)
        if isinstance(output, pd.DataFrame):
            logging.info(
                f"{func.__name__} function returned dataframe with shape: {output.shape}, columns: {', '.join(output.columns)}"
            )
        else:
            logging.info(f"{func.__name__} function returned result: {output}")
        return output

    return inner


def download_kaggle_dataset() -> None:
    kaggle.api.dataset_download_files(
        dataset=config.AVIATION_DATA_API,
        path=config.KAGGLE_DATASET_DOWNLOAD_PATH,
        unzip=False,
        quiet=False,
        force=False,
    )


def unzip_file() -> None:
    with zipfile.ZipFile(config.KAGGLE_DATASET_ZIPPED_FILENAME, "r") as zip_ref:
        zip_ref.extractall(config.KAGGLE_DATASET_EXTRACT_PATH)


@logger_df
def load_dataset() -> pd.DataFrame:
    path = config.KAGGLE_DATASET_EXTRACTED_FILENAME
    if not os.path.isfile(path):
        download_kaggle_dataset()
        unzip_file()
    df = pd.read_csv(
        config.KAGGLE_DATASET_EXTRACTED_FILENAME,
        parse_dates=["Event.Date", "Publication.Date"],
        encoding="cp1252",
        low_memory=False,
    )
    return df


@logger_df
def save_to_csv(df: pd.DataFrame) -> None:
    df.to_csv(config.INTERIM_DIRECTORY, index=False)


@logger_df
@pa.check_types
def column_name_replacement(
    df, what_to_replace: str, replacement: str
) -> DataFrame[InputSchema]:
    """Replaces symbols, letters in all column names with a given string"""
    df.columns = df.columns.str.replace(what_to_replace, replacement, regex=True)
    return df


@logger_df
def separate_city_and_state(df: pd.DataFrame) -> pd.DataFrame:
    """Separates city and state"""
    df_city_state = df["Location"].str.split(",", n=1, expand=True)
    dfr = df.assign(City=df_city_state[0], State=df_city_state[1])
    return dfr


@logger_df
def create_year_and_month_column_from_date(df: pd.DataFrame) -> pd.DataFrame:
    """Separates year and month from accident date"""
    df = df.assign(Event_year=df["Event_Date"].dt.year)
    df = df.assign(Event_month=df["Event_Date"].dt.month)
    return df


def filter_by_period(
    df, column_name: str, start_year: int, end_year: int
) -> pd.DataFrame:
    return df[(df[column_name] >= start_year) & (df[column_name] <= end_year)]


@logger_df
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


@logger_df
def get_min_max_sum_death_injuries_by_injury_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Returns Injury severity groups and min, max and sum death accidents"""
    grouped_by_severity = df.groupby("Injury_Severity")["Total_Fatal_Injuries"].agg(
        ["min", "max", "sum"]
    )
    return grouped_by_severity


@logger_df
def get_incidents_per_year(df):
    accidents_per_year = (
        df.groupby(["Event_year"], as_index=False)["Event_Id"]
        .count()
        .rename(columns={"Event_Id": "Count"})
    )
    return accidents_per_year


@logger_df
def add_sum_of_total_people_in_accident(df: pd.DataFrame) -> pd.DataFrame:
    sum_of_people = df[
        [
            "Total_Fatal_Injuries",
            "Total_Serious_Injuries",
            "Total_Minor_Injuries",
            "Total_Uninjured",
        ]
    ].agg(["sum"], axis=1)
    return df.assign(Total_people_in_accident=sum_of_people)


@logger_df
def timedelta_between_accident_and_publication(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates difference in days between publication date and event date"""
    timedelta = (df["Publication_Date"] - df["Event_Date"]).dt.days
    df_with_timedelta = df.assign(Time_between_publication_and_event=timedelta)
    return df_with_timedelta


@logger_df
def get_most_freq_airplane_make_engine_type_and_flight_purpose(
    top_makes: pd.DataFrame, top_purpose: pd.DataFrame, top_engine_type: pd.DataFrame
) -> pd.DataFrame:
    """Function returns manufacturer and type of airplanes that participates in accidents most frequently"""
    df = pd.concat(
        [top_makes, top_purpose, top_engine_type],
        axis=1,
    ).nlargest(1, ["Max_make", "Max_purpose", "Max_type"])
    return df


def get_flight_purpose_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Function return flight purpose with amounts that get into accidents"""
    return (
        df["Purpose_of_flight"]
        .value_counts()
        .reset_index()
        .rename(
            columns={"index": "Purpose_of_flight", "Purpose_of_flight": "Max_purpose"}
        )
    )


def get_airplane_make_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Function return airplane makes with amounts that get into accidents"""
    df["Make"] = df["Make"].str.lower()
    return (
        df["Make"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Make", "Make": "Max_make"})
    )


def get_airplane_engine_type_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Function return airplane engine type with amounts that get into accidents"""
    return (
        df["Engine_Type"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Engine_Type", "Engine_Type": "Max_type"})
    )


def accident_statistics_by_airplane_make_engine_flight_purpose(
    df: pd.DataFrame,
) -> pd.DataFrame:
    flight_purpose_statistics = get_flight_purpose_statistics(df)
    airplane_make_statistics = get_airplane_make_statistics(df)
    airplane_engine_type_statistics = get_airplane_engine_type_statistics(df)
    result = get_most_freq_airplane_make_engine_type_and_flight_purpose(
        airplane_make_statistics,
        flight_purpose_statistics,
        airplane_engine_type_statistics,
    )
    return result


def plot_accidents_amount_by_state(df: pd.DataFrame) -> None:
    """plotting events by states count"""
    plt.figure(1)
    sns.countplot(
        y="State",
        data=df[(df["Country"] == "United States") & ((df["State"].str.len() <= 3))],
    )


def plot_time_between_publication_and_event(df: pd.DataFrame) -> None:
    """plotting histogram of time between publication and event"""
    plt.figure(2)
    df["Time_between_publication_and_event"].plot.hist(
        bins=12, legend=True, xlim=(0, 6000)
    )


def plot_accidents_per_year(df_accidents_per_year: pd.DataFrame) -> None:
    """plotting histogram of accidents per year"""
    plt.figure(3)
    sns.lineplot(data=df_accidents_per_year, x="Event_year", y="Count", color="#2990EA")


@logger_df
def preprocese_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df_processed = (
        df.pipe(column_name_replacement, ".", "_")
        .pipe(separate_city_and_state)
        .pipe(create_year_and_month_column_from_date)
        .pipe(remove_symbols_and_digits_from_column, "Injury_Severity")
        .pipe(timedelta_between_accident_and_publication)
        .pipe(add_sum_of_total_people_in_accident)
    )
    return df_processed


@logger_df
def add_data_from_weatherbit_api(df: pd.DataFrame) -> pd.DataFrame:
    """Reads data (temperature by day) from Weatherbit.io regarding accident data"""
    temperatures = []
    df = df.tail(5)
    for row in df.itertuples():
        params = build_weatherbit_api_params(row)
        response = api_request_weatherbit_api(params)
        temperatures.append(response)
    return df.assign(Temperatures_accident_day=temperatures)


def build_weatherbit_api_params(row) -> dict:
    start_date = row.Event_Date.date()
    end_date = start_date + timedelta(days=1)
    params = {
        "key": [config.WEATHERBIT_API_KEY],
        "start_date": start_date,
        "end_date": end_date,
        "city": row.City,
    }
    return params


def api_request_weatherbit_api(parameters):
    method = "ping"
    api_base = "https://api.weatherbit.io/v2.0/history/daily?"
    try:
        api_result = requests.get(api_base + method, parameters)
        response = api_result.json()
    except ValueError:
        return np.nan
    return response["data"][0]["temp"]


if __name__ == "__main__":
    logging.basicConfig(
        filename=config.LOGGER_FILENAME,
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
    )
    df = load_dataset()
    df_processed = preprocese_dataset(df)

    death_injuries_statistics = get_min_max_sum_death_injuries_by_injury_groups(
        df_processed
    )
    incidents_per_year = get_incidents_per_year(df_processed)

    accidents_by_period = get_accident_amount_by_period(
        df_processed, "Event_year", 2020, 2023
    )

    accident_statistics_by_airplane_make_engine_flight_purpose(df_processed)

    # df_with_external_data = add_data_from_weatherbit_api(df_processed)

    plot_accidents_per_year(incidents_per_year)
    plt.show()

""" Future Functions for final data representation"""
# plot_time_between_publication_and_event(df_processed)
# save_to_csv(df_temp)
# create_visualization = visualize(df)
# final_report_df = create_report(df)
# df.to_excel("sdfsdf.xlsx")
