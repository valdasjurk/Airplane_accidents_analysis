# Source: https://www.kaggle.com/datasets/khsamaha/aviation-accident-database-synopses

import kaggle
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series

# import seaborn as sns
# import matplotlib.pyplot as plt

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


def save_to_csv(df):
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
    # """ plotting events by states count"""
    # sns.countplot(
    #     y="State",
    #     data=dfr[dfr["Country"] == "United States"],
    # )
    # plt.show()
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
    """ plotting results in histogram """
    fig = timedelta.plot.hist(bins=12, legend=True, xlim=(0, 6000))
    fig.figure.savefig("output/Timedelta.png")
    return df.assign(Time_between_publication_and_event=timedelta)


def preprocese_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df_processed = (
        df.pipe(column_name_replacement, ".", "_")
        .pipe(separate_city_and_state)
        .pipe(create_year_and_month_column_from_date)
        .pipe(remove_symbols_and_digits_from_column, "Injury_Severity")
    )
    return df_processed


if __name__ == "__main__":
    df = read_dataset()
    df_processed = preprocese_dataset(df)

    death_injuries_statistics = get_min_max_sum_death_injuries_by_injury_groups(
        df_processed
    )
    accidents_by_period = get_accident_amount_by_period(
        df_processed, "Event_year", 2020, 2023
    )


""" Future Functions for final data representation"""
# create_visualization = visualize(df)
# final_report_df = create_report(df)
# df.to_excel("sdfsdf.xlsx")
