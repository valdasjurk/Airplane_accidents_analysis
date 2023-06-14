import kaggle
import zipfile
import os
import config
import pandas as pd
from utils import logger_df


def download_kaggle_dataset() -> None:
    kaggle.api.dataset_download_files(
        dataset=config.AVIATION_DATA_API,
        path=config.KAGGLE_DATASET_DOWNLOAD_PATH,
        unzip=False,
        quiet=False,
        force=False,
    )


def unzip_file() -> None:
    with zipfile.ZipFile(config.KAGGLE_DATASET_ZIPPED_FILENAME, "r") as zip_ref:
        zip_ref.extractall(config.KAGGLE_DATASET_EXTRACT_PATH)


@logger_df
def load_dataset() -> pd.DataFrame:
    path = config.KAGGLE_DATASET_EXTRACTED_FILENAME
    if not os.path.isfile(path):
        download_kaggle_dataset()
        unzip_file()
    df = pd.read_csv(
        config.KAGGLE_DATASET_EXTRACTED_FILENAME,
        parse_dates=["Event.Date", "Publication.Date"],
        infer_datetime_format=True,
        encoding="cp1252",
        low_memory=False,
    )
    return df


def save_to_csv(df: pd.DataFrame, path=config.PROCESSED_DIRECTORY, add="") -> None:
    path_final = path + add
    df.to_csv(path_final, index=False)
