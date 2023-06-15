import logging
import pandas as pd


def logger_df(func):
    def inner(*args, **kwargs):
        logging.info(f"Started executing function: {func.__name__}")
        output = func(*args, **kwargs)
        if isinstance(output, pd.DataFrame):
            logging.info(
                f"{func.__name__} function returned dataframe with shape: {output.shape}, columns: {', '.join(output.columns)}"
            )
        else:
            logging.info(f"{func.__name__} function returned result: {output}")
        return output

    return inner
