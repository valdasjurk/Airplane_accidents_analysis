# Source: https://www.kaggle.com/datasets/khsamaha/aviation-accident-database-synopses

import json
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
        logging.info(
            f"{func.__name__} function returned dataframe with shape: {output.shape}"
        )
        logging.info(
            f"{func.__name__} function returned dataframe with columns: {', '.join(output.columns)}"
        )
        return output

    return inner


def download_dataset() -> None:
    kaggle.api.dataset_download_files(
        dataset="khsamaha/aviation-accident-database-synopses",
        path="data",
        unzip=True,
        quiet=False,
        force=False,
    )


@logger_df
def read_dataset() -> pd.DataFrame:
    df = pd.read_csv(
        config.RAW_DATA_DIRECTORY,
        parse_dates=["Event.Date", "Publication.Date"],
        encoding="cp1252",
        low_memory=False,
        # na_values=" ",
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
    df.columns = df.columns.str.replace(what_to_replace, replacement)
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


def get_accident_amount_by_period(
    df, column_name: str, start_year: int, end_year: int
) -> int:
    """Returns accident amount by a given period"""
    logging.info(
        f"Started calculate accident amount by period {start_year} - {end_year}"
    )
    df_by_condition = df[
        (df[column_name] >= start_year) & (df[column_name] <= end_year)
    ]
    amount_of_accidents = len(df_by_condition)
    logging.info(f"Amount of accidents: {amount_of_accidents}")
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
    logging.info(f"severity groups statistics:\n {grouped_by_severity}")
    return grouped_by_severity


@logger_df
def timedelta_between_accident_and_publication(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates difference in days between publication date and event date"""
    timedelta = (df["Publication_Date"] - df["Event_Date"]).dt.days
    df_with_timedelta = df.assign(Time_between_publication_and_event=timedelta)
    return df_with_timedelta


def get_most_freq_airplane_make_and_type(
    top_makes: pd.Series, top_purpose: pd.Series
) -> pd.DataFrame:
    """Function returns manufacturer and type of airplanes that participates in accidents most frequently"""
    logging.info(
        " Started calculate manufacturer and type of airplanes that participates in accidents most frequently..."
    )
    most_freq_make = top_makes.nlargest(1)
    most_freq_purpose = top_purpose.nlargest(1)
    df_most_freq = pd.concat([most_freq_make, most_freq_purpose], axis=0).reset_index()
    df_most_freq.columns = ["Statistics object", "Count"]
    logging.info(f"Make and purpose statistics:\n {df_most_freq}")
    return df_most_freq


def get_flight_purpose_statistics(df: pd.DataFrame) -> pd.Series:
    """Function flight purpose statistics"""
    return df["Purpose_of_flight"].value_counts()


def get_airplane_make_statistics(df: pd.DataFrame) -> pd.Series:
    """Returns top 10 airplane makes that gets into the accidents"""
    df["Make"] = df["Make"].str.lower()
    return df["Make"].value_counts()


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


@logger_df
def preprocese_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df_processed = (
        df.pipe(column_name_replacement, ".", "_")
        .pipe(separate_city_and_state)
        .pipe(create_year_and_month_column_from_date)
        .pipe(remove_symbols_and_digits_from_column, "Injury_Severity")
        .pipe(timedelta_between_accident_and_publication)
    )
    return df_processed


def add_data_from_weatherbit_api(df: pd.DataFrame) -> pd.DataFrame:
    """Reads data (temperature by day) from Weatherbit.io regarding accident data"""
    logging.info("Started adding data from weatherbit...")
    temperatures = []
    df = df.tail(5)
    for index, row in df.iterrows():
        start_date = row["Event_Date"].date()
        end_date = start_date + timedelta(days=1)
        params = {
            "key": [config.WEATHERBIT_API_KEY],
            "start_date": start_date,
            "end_date": end_date,
            "city": row["City"],
        }
        response_dict = api_request_weatherbit_api(params)
        for key in response_dict["data"]:
            temperatures.append(key["temp"])
    logging.info("Finished adding data from weatherbit!")
    return df.assign(Temperatures_accident_day=temperatures)


def api_request_weatherbit_api(parameters):
    method = "ping"
    api_base = "https://api.weatherbit.io/v2.0/history/daily?"
    try:
        api_result = requests.get(api_base + method, parameters)
        response = api_result.text
        response_dict = json.loads(response)
    except ValueError:
        response_dict = {"data": {"temp": np.nan}}
    return response_dict


def dataframe_logging(df):
    logging.info(f"Column names: {', '.join(df.columns)}")
    logging.info(f"Dataframe shape (rows, columns): {df.shape}")


if __name__ == "__main__":
    logging.basicConfig(
        filename=config.LOGGER_FILENAME,
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
    )

    df = read_dataset()
    df_processed = preprocese_dataset(df)

    death_injuries_statistics = get_min_max_sum_death_injuries_by_injury_groups(
        df_processed
    )
    accidents_by_period = get_accident_amount_by_period(
        df_processed, "Event_year", 2020, 2023
    )

    # df_with_external_data = add_data_from_weatherbit_api(df_processed)
    # print(df_with_external_data.head())

    flight_purpose_statistics = get_flight_purpose_statistics(df_processed)
    airplane_make_statistics = get_airplane_make_statistics(df_processed)
    get_most_freq_airplane_make_and_type(
        airplane_make_statistics, flight_purpose_statistics
    )


""" Future Functions for final data representation"""
# plot_time_between_publication_and_event(df_processed)
# save_to_csv(df_temp)
# create_visualization = visualize(df)
# final_report_df = create_report(df)
# df.to_excel("sdfsdf.xlsx")
