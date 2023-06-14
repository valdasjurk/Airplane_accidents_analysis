# Source: https://www.kaggle.com/datasets/khsamaha/aviation-accident-database-synopses


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from load_and_save_airplane_accidents_dataset import load_dataset
from preprocesse_dataset import preprocese_dataset
from utils import logger_df

# from weatherbit_api import add_data_from_weatherbit_api


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


@logger_df
def get_min_max_sum_death_injuries_by_injury_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Returns Injury severity groups and min, max and sum death accidents"""
    grouped_by_severity = df.groupby("Injury_Severity")["Total_Fatal_Injuries"].agg(
        ["min", "max", "sum"]
    )
    return grouped_by_severity


@logger_df
def get_incidents_per_year(df: pd.DataFrame) -> pd.DataFrame:
    accidents_per_year = (
        df.groupby(["Event_year"], as_index=False)["Event_Id"]
        .count()
        .rename(columns={"Event_Id": "Count"})
    )
    return accidents_per_year


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


@logger_df
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


@logger_df
def get_airplane_make_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Function return airplane makes with amounts that get into accidents"""
    df["Make"] = df["Make"].str.lower()
    return (
        df["Make"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Make", "Make": "Max_make"})
    )


@logger_df
def get_airplane_engine_type_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Function return airplane engine type with amounts that get into accidents"""
    return (
        df["Engine_Type"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Engine_Type", "Engine_Type": "Max_type"})
    )


@logger_df
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


def plot_accidents_amount_by_state(df: pd.DataFrame) -> plt.Axes:
    """plotting events by states count"""
    plot = sns.countplot(
        y="State",
        data=df[(df["Country"] == "United States") & ((df["State"].str.len() <= 3))],
    )
    return plot


def plot_time_between_publication_and_event(df: pd.DataFrame) -> plt.Axes:
    """plotting histogram of time between publication and event"""
    plot = df["Time_between_publication_and_event"].plot.hist(
        bins=12, legend=True, xlim=(0, 6000)
    )
    return plot


def plot_accidents_per_year(df_accidents_per_year: pd.DataFrame) -> plt.Axes:
    """plotting histogram of accidents per year"""
    plot = sns.lineplot(
        data=df_accidents_per_year, x="Event_year", y="Count", color="#2990EA"
    )
    return plot


if __name__ == "__main__":
    df = load_dataset()
    df_processed = preprocese_dataset(df)

    death_injuries_statistics = get_min_max_sum_death_injuries_by_injury_groups(
        df_processed
    )
    incidents_per_year = get_incidents_per_year(df_processed)
    print(incidents_per_year)

    accidents_by_period = get_accident_amount_by_period(
        df_processed, "Event_year", 2020, 2023
    )

    accident_statistics_by_airplane_make_engine_flight_purpose(df_processed)

    # df_with_external_data = add_data_from_weatherbit_api(df_processed)
