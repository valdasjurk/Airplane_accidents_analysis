from utils import logger_df
import requests
import pandas as pd
from datetime import timedelta
import config
import numpy as np


@logger_df
def add_data_from_weatherbit_api(df: pd.DataFrame, year=2022, month=12) -> pd.DataFrame:
    """Reads data (temperature by day) from Weatherbit.io regarding accident data.
    Currently it supports only one month of the year due to limited API access.
    """
    temperatures = []
    df = df.loc[(df["Event_year"] == year) & (df["Event_month"] == month)]
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
