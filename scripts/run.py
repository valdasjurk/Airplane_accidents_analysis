from general import (
    download_dataset,
    preprocese_dataset,
    read_dataset,
    get_min_max_sum_death_injuries_by_injury_groups,
    get_accident_amount_by_period,
)


def download_data():
    download_dataset()


def preprocesse():
    df = read_dataset()
    df_processed = preprocese_dataset(df)
    return df_processed


def get_results():
    df = preprocesse()
    death_injuries_statistics = get_min_max_sum_death_injuries_by_injury_groups(df)
    accidents_by_period = get_accident_amount_by_period(df, "Event_year", 2020, 2023)
    print("Death injuries statistics: ", death_injuries_statistics)
    print("Accidents by period: ", accidents_by_period)
