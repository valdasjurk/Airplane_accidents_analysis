import config
import pandas as pd


def save_to_csv(df: pd.DataFrame, path=config.INTERIM_DIRECTORY) -> None:
    df.to_csv(path, index=False)
