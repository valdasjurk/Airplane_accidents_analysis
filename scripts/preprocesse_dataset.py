import pandas as pd
import pandera as pa
from pandera.typing import DataFrame
from panderas_schemas import Airplanes_dataset_InputSchema
from utils import logger_df
import logging
import config

logging.basicConfig(
    filename=config.LOGGER_FILENAME,
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
)


@logger_df
@pa.check_types
def _column_name_replacement(
    df, what_to_replace: str, replacement: str
) -> DataFrame[Airplanes_dataset_InputSchema]:
    """Replaces symbols, letters in all column names with a given string"""
    df.columns = df.columns.str.replace(what_to_replace, replacement, regex=True)
    return df


@logger_df
def _separate_city_and_state(df: pd.DataFrame) -> pd.DataFrame:
    """Separates city and state"""
    df_city_state = df["Location"].str.split(",", n=1, expand=True)
    dfr = df.assign(City=df_city_state[0], State=df_city_state[1])
    return dfr


@logger_df
def _create_year_and_month_column_from_date(df: pd.DataFrame) -> pd.DataFrame:
    """Separates year and month from accident date"""
    df = df.assign(Event_year=df["Event_Date"].dt.year)
    df = df.assign(Event_month=df["Event_Date"].dt.month)
    return df


@logger_df
def _remove_symbols_and_digits_from_column(
    df: pd.DataFrame, column_name: str
) -> pd.DataFrame:
    df[column_name] = df[column_name].str.replace(r"\W", "", regex=True)
    df[column_name] = df[column_name].str.replace(r"\d+", "", regex=True)
    return df


@logger_df
def _timedelta_between_accident_and_publication(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates difference in days between publication date and event date"""
    timedelta = (df["Publication_Date"] - df["Event_Date"]).dt.days
    df_with_timedelta = df.assign(Time_between_publication_and_event=timedelta)
    return df_with_timedelta


@logger_df
def _add_sum_of_total_people_in_accident(df: pd.DataFrame) -> pd.DataFrame:
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
def preprocese_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df_processed = (
        df.pipe(_column_name_replacement, ".", "_")
        .pipe(_separate_city_and_state)
        .pipe(_create_year_and_month_column_from_date)
        .pipe(_remove_symbols_and_digits_from_column, "Injury_Severity")
        .pipe(_timedelta_between_accident_and_publication)
        .pipe(_add_sum_of_total_people_in_accident)
    )
    return df_processed
