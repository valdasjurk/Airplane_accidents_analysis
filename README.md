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
Get min, max and sum death accidents by injury severity groups.
```
python run.py --get_injury_statistics
```
Get accidents per year data from 1982 to 2022.
```
python run.py --get_accidents_sum_by_year
```
Get visualisation of airplane accidents amount by US state. Command goes together with "how" argument which has options to show or save the figure.
```
python run.py --visualise_accidents_amount_by_state --how ["show", "save"]
```
Get visualisation of time in days between airplane accident date and publication date. Command goes together with "how" argument which has options to show or save the figure.
```
python run.py --visualise_time_between_publication_and_event --how ["show", "save"]
```
Get visualisation of worldwide airplane accidents per year since 1982 to 2022. Command goes together with "how" argument which has options to show or save the figure.
```
python run.py --visualise_accidents_per_year --how ["show", "save"]
```
## Analysis questions

1. Column name replacement to make them readable.
2. Separate City from State and create new columns with them
3. Get year and month from Event_date. New columns: year, month.
4. Calculate time difference in days between accident date and pusblishing date. Results are stored in a new column.
5. Amount of accidents within a given year interval. Calculate percentage of all accidents.
6. Calculate which type of airplane makes, engines and airplane types fail the most.
7. Groupby "Injury_serverity". Get max, min, mean of injured persons by groups.
8. Calculate total airplane accidents per year.
9. Visualisations of US states accident statistics, histogram of time between accident and publication, accidents per year statistics.
10. Add data from external api about weather conditions during accident day (Due to api restrictions, full dataset cannot be covered).
