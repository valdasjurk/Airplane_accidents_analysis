# Airplane crash data analysis

Source: https://www.kaggle.com/datasets/khsamaha/aviation-accident-database-synopses

```
project/
├── data/
│ ├── raw/
│ ├── processed/
│ └── interim/
├── notebooks/
├── scripts/
├── output/
├── README.md
└── requirements.txt
```
## Installation

Install required modules:
```
pip install -r requirements.txt
```

Function only downloads dataset to data/raw
```
python run.py --load_dataset
```
Preprocesse and store dataset to data/interim. Function downloads dataset if it does not exist.
```
python run.py --prepare_and_save_data
```
Get death injuries statistics, accidents by period
```
python run.py --get_accidents_by_period
```
Get airplanes makes, engine type and flight type that gets into accidents most frequently.
```
python run.py --get_statistics_airplane_make_engine_flight_purpose
```

## Analysis questions

1. Deal with missing values.
2. Column name replacement
3. Separate City from State and create new columns with them
4. Get year and month from Event_date. New columns: year, month.
5. Calculate time difference in days between accident date and pusblishing date. Results are stored in a new column.
6. Amount of accidents within a given year interval. Calculate percentage of all accidents.
7. Calculate which type of airplane makes, engines and airplane types fail the most.
8. Groupby "Injury_serverity". Get max, min, mean of injured persons by groups.
9. How many accidents each boarding phase has? create Series with phase and accidents as .int
10. Return plots about US states accident statistics, histogram of time between accident and publication, accidents per year statistics.
11. Add data from external api about weather conditions during accident day (Due to api restrictions, full dataset cannot be covered).
