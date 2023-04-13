import kaggle
import pandas as pd


DIRECTORY = "data/raw/AviationData.csv"


def download_dataset():
    kaggle.api.dataset_download_files(
        dataset="khsamaha/aviation-accident-database-synopses",
        path="data",
        unzip=True,
        quiet=False,
        force=False,
    )


def read_dataset():
    df = pd.read_csv(DIRECTORY, encoding="cp1252")
    df = df.drop_duplicates()
    return df


def reasonable_masks(df):
    df["Event.Date"] = pd.to_datetime(df["Event.Date"])
    year_filtering = df[df["Event.Date"].dt.year > 2020]
    print(year_filtering)
    city_name_mask = df["Location"].str.contains("SEASIDE")
    print(city_name_mask)
    df_by_city_name = df[df["Location"] == "SEASIDE HEIGHTS, NJ"]
    print(df_by_city_name)
    df_outside_usa = df[df["Country"] != "United States"]
    print(df_outside_usa)
    no_of_engines_mask = df.query("Number_of_Engines > 1")
    print(no_of_engines_mask)
    make_mask = df["Make"].isin(["Stinson", "Cessna", "Piper"])
    print(make_mask)


# df.groupby([df['birthdate'].dt.year, df['birthdate'].dt.month]).agg({'count'})

if __name__ == "__main__":
    # download_dataset()
    df = read_dataset()

    reasonable_masks(df)
