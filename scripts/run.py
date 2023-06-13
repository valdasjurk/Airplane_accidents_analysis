from general import (
    load_dataset,
    preprocese_dataset,
    get_accident_amount_by_period,
    save_to_csv,
    accident_statistics_by_airplane_make_engine_flight_purpose,
    plot_accidents_amount_by_state,
    plot_time_between_publication_and_event,
    plot_accidents_per_year,
)
import argparse
import config
import pandas as pd
import matplotlib.pyplot as plt


def prepare_and_save_data():
    df = load_dataset()
    df_processed = preprocese_dataset(df)
    save_to_csv(df_processed)


def load_preprocessed_data(path=config.INTERIM_DIRECTORY):
    return pd.read_csv(path, low_memory=False)


def get_accidents_by_period(df, start, end):
    accidents_by_period = get_accident_amount_by_period(df, "Event_year", start, end)
    return accidents_by_period


def plot_show_or_save(plot, how, filename="output/graphs/output.jpg"):
    fig = plot.get_figure()
    if how == "show":
        plt.show()
    elif how == "save":
        fig.savefig(filename)


if __name__ == "__main__":
    # Get our arguments from the user
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--load_dataset", help="Download kaggle dataset", action="store_true"
    )
    parser.add_argument(
        "--prepare_and_save_data",
        help="Datased is preprocessed and saved to data/interim",
        action="store_true",
    )
    parser.add_argument(
        "--get_accidents_by_period",
        help="Calculate amount of accidents per given period. Requires prepared data. See --load_and_save_data",
        action="store_true",
    )
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)

    parser.add_argument(
        "--get_statistics_airplane_make_engine_flight_purpose",
        help="Calculate most frequent airplane make, engine type and flight purpose that gets into accidents",
        action="store_true",
    )

    parser.add_argument("--visualise_accidents_amount_by_state", action="store_true")
    parser.add_argument(
        "--visualise_time_between_publication_and_event", action="store_true"
    )
    parser.add_argument("--visualise_accidents_per_year", action="store_true")
    parser.add_argument("--how", type=str)
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
        df = pd.DataFrame()
        try:
            df = load_preprocessed_data()
        except FileNotFoundError:
            print("No prepared data found. Did you run --prepare_and_save_data ?")
        results = accident_statistics_by_airplane_make_engine_flight_purpose(df)
        print(results)

    if args.visualise_accidents_amount_by_state:
        df = pd.DataFrame()
        try:
            df = load_preprocessed_data()
        except FileNotFoundError:
            print("No prepared data found. Did you run --prepare_and_save_data ?")
        plot = plot_accidents_amount_by_state(df)
        fig = plot.get_figure()
        plot_show_or_save(fig, args.how, filename="output/graphs/state.jpg")

    if args.visualise_time_between_publication_and_event:
        df = pd.DataFrame()
        try:
            df = load_preprocessed_data()
        except FileNotFoundError:
            print("No prepared data found. Did you run --prepare_and_save_data ?")
        plot = plot_time_between_publication_and_event(df)
        plot_show_or_save(plot, args.how, filename="output/graphs/timedelta.jpg")

    if args.visualise_accidents_per_year:
        df = pd.DataFrame()
        try:
            df = load_preprocessed_data()
        except FileNotFoundError:
            print("No prepared data found. Did you run --prepare_and_save_data ?")
        plot = plot_accidents_per_year(df)
        plot_show_or_save(
            plot, args.how, filename="output/graphs/accidents_per_year.jpg"
        )
