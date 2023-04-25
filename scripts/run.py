from general import (
    download_dataset,
    preprocese_dataset,
    read_dataset,
    get_min_max_sum_death_injuries_by_injury_groups,
    get_accident_amount_by_period,
    save_to_csv,
)
import argparse


def download_data():
    download_dataset()


def preprocesse():
    df = read_dataset()
    df_processed = preprocese_dataset(df)
    save_to_csv(df_processed)
    return df_processed


def get_results():
    df = preprocesse()
    death_injuries_statistics = get_min_max_sum_death_injuries_by_injury_groups(df)
    accidents_by_period = get_accident_amount_by_period(df, "Event_year", 2020, 2023)
    print("Death injuries statistics: ", death_injuries_statistics)
    print("Accidents by period: ", accidents_by_period)


if __name__ == "__main__":
    # Get our arguments from the user
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--download_data", help="Download kaggle dataset", action="store_true"
    )
    parser.add_argument(
        "--preprocesse",
        help="Function returns preprocessed dataframe",
        action="store_true",
    )
    parser.add_argument(
        "--get_results",
        help="Get statistics and amount of accidents",
        action="store_true",
    )
    args = parser.parse_args()

    if args.download_data:
        download_data()

    if args.preprocesse:
        preprocesse()

    if args.get_results:
        get_results()
