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

Analysis questions: 
1. Deal with missing values.
2. Column name replacement
3. Separate City from State and create new columns with them
4. Get year and month from Event_date. New columns: year, month.
5. Amount of accidents within a given year interval. Calculate percentage of all accidents.
6. Time delta between accident date and pusblishing date. Store results in a new column. 
7. Which type of engines fail the most. 
8. Groupby "Injury_serverity". Get max, min, mean of injured persons by groups.
9. How many cities  and states are there?. Int value.
10. How many accidents each boarding phase has? create Series with phase and accidents as .int
