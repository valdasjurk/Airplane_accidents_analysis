from general import (
    load_dataset,
    preprocese_dataset,
    get_accident_amount_by_period,
    save_to_csv,
    accident_statistics_by_airplane_make_engine_flight_purpose,
)
import argparse
import config
import pandas as pd


def prepare_and_save_data():
    df = load_dataset()
    df_processed = preprocese_dataset(df)
    save_to_csv(df_processed)


def load_preprocessed_data(path=config.INTERIM_DIRECTORY):
    return pd.read_csv(path, low_memory=False)


def get_accidents_by_period(df, start, end):
    accidents_by_period = get_accident_amount_by_period(df, "Event_year", start, end)
    return accidents_by_period


if __name__ == "__main__":
    # Get our arguments from the user
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--load_dataset", help="Download kaggle dataset", action="store_true"
    )
    parser.add_argument(
        "--prepare_and_save_data",
        help="Function returns preprocessed dataframe",
        action="store_true",
    )
    parser.add_argument(
        "--get_accidents_by_period",
        help="Get statistics and amount of accidents. Requires prepared data. See --load_and_save_data",
        action="store_true",
    )
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)

    parser.add_argument(
        "--get_statistics_airplane_make_engine_flight_purpose",
        help="Function returns dataframe with airplane make, engine type and flight purpose that gets into accidents most frequently",
        action="store_true",
    )
    args = parser.parse_args()

    if args.load_dataset:
        load_dataset()

    if args.prepare_and_save_data:
        prepare_and_save_data()

    if args.get_accidents_by_period:
        try:
            df = load_preprocessed_data()
            results = get_accidents_by_period(df, args.start, args.end)
            print(results)
        except FileNotFoundError:
            print("No prepared data found. Did you run --prepare_and_save_data ?")

    if args.get_statistics_airplane_make_engine_flight_purpose:
        try:
            df = load_preprocessed_data()
            results = accident_statistics_by_airplane_make_engine_flight_purpose(df)
            print(results)
        except FileNotFoundError:
            print("No prepared data found. Did you run --prepare_and_save_data ?")
