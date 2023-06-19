from preprocesse_dataset import (
    _create_year_and_month_column_from_date,
    _timedelta_between_accident_and_publication,
    _remove_symbols_and_digits_from_column,
    _separate_city_and_state,
    _column_name_replacement,
)
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
        _create_year_and_month_column_from_date(df), df_expected
    )


def test_timedelta_in_days():
    df = pd.DataFrame(
        {
            "Event_Date": [pd.Timestamp(2022, 1, 1), pd.Timestamp(2023, 2, 2)],
            "Publication_Date": [pd.Timestamp(2022, 1, 2), pd.Timestamp(2023, 2, 5)],
        }
    )
    df_expected = pd.DataFrame(
        {
            "Event_Date": [pd.Timestamp(2022, 1, 1), pd.Timestamp(2023, 2, 2)],
            "Publication_Date": [pd.Timestamp(2022, 1, 2), pd.Timestamp(2023, 2, 5)],
            "Time_between_publication_and_event": [1, 3],
        }
    )
    pd.testing.assert_frame_equal(
        _timedelta_between_accident_and_publication(df), df_expected
    )


def test_remove_symbols_and_digits_from_column_values():
    df = pd.DataFrame(
        {
            "Injury": ["Fatal2", "Serious(1)"],
        }
    )
    df_expected = pd.DataFrame(
        {
            "Injury": ["Fatal", "Serious"],
        }
    )
    pd.testing.assert_frame_equal(
        _remove_symbols_and_digits_from_column(df, "Injury"), df_expected
    )


def test_separate_city_and_state():
    df = pd.DataFrame(
        {"Location": ["Seaside Heights,NJ"], "Country": ["United States"]}
    )
    df_expected = pd.DataFrame(
        {
            "Location": ["Seaside Heights,NJ"],
            "Country": ["United States"],
            "City": ["Seaside Heights"],
            "State": ["NJ"],
        }
    )
    pd.testing.assert_frame_equal(_separate_city_and_state(df), df_expected)


def test_column_name_replacement():
    df = pd.DataFrame(
        {
            "Injury.total": [1, 2],
        }
    )
    df_expected = pd.DataFrame(
        {
            "Injury_total": [1, 2],
        }
    )
    pd.testing.assert_frame_equal(_column_name_replacement(df, ".", "_"), df_expected)
