from general import create_year_and_month_column_from_date
import pandas as pd


def test_get_year_and_month_from_date():
    df = pd.DataFrame(
        {"Event_Date": [pd.Timestamp(2022, 1, 1), pd.Timestamp(2023, 2, 2)]}
    )
    df_expected = pd.DataFrame(
        {
            "Event_Date": [pd.Timestamp(2022, 1, 1), pd.Timestamp(2023, 2, 2)],
            "Event_year": [2022, 2023],
            "Event_month": [1, 2],
        }
    )
    pd.testing.assert_frame_equal(
        create_year_and_month_column_from_date(df), df_expected
    )
