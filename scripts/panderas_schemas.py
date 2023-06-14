from pandera.typing import Series
import pandera as pa
import pandas as pd


class Airplanes_dataset_InputSchema(pa.SchemaModel):
    Event_Id: Series[str]
    Investigation_Type: Series[str]
    Accident_Number: Series[str]
    Event_Date: Series[pd.DatetimeTZDtype] = pa.Field(
        dtype_kwargs={"unit": "ns", "tz": "EST"}
    )
    Location: Series[str] = pa.Field(nullable=True)
    Country: Series[str] = pa.Field(nullable=True)
    Latitude: Series[str] = pa.Field(nullable=True)
    Longitude: Series[str] = pa.Field(nullable=True)
    Airport_Code: Series[str] = pa.Field(nullable=True)
    Airport_Name: Series[str] = pa.Field(nullable=True)
    Injury_Severity: Series[str] = pa.Field(nullable=True)
    Aircraft_damage: Series[str] = pa.Field(nullable=True)
    Aircraft_Category: Series[str] = pa.Field(nullable=True)
    Registration_Number: Series[str] = pa.Field(nullable=True)
    Make: Series[str] = pa.Field(nullable=True)
    Model: Series[str] = pa.Field(nullable=True)
    Amateur_Built: Series[str] = pa.Field(nullable=True)
    Number_of_Engines: Series[float] = pa.Field(nullable=True)
    Engine_Type: Series[str] = pa.Field(nullable=True)
    FAR_Description: Series[str] = pa.Field(nullable=True)
    Schedule: Series[str] = pa.Field(nullable=True)
    Purpose_of_flight: Series[str] = pa.Field(nullable=True)
    Air_carrier: Series[str] = pa.Field(nullable=True)
    Total_Fatal_Injuries: Series[float] = pa.Field(nullable=True)
    Total_Serious_Injuries: Series[float] = pa.Field(nullable=True)
    Total_Minor_Injuries: Series[float] = pa.Field(nullable=True)
    Total_Uninjured: Series[float] = pa.Field(nullable=True)
    Weather_Condition: Series[str] = pa.Field(nullable=True)
    Broad_phase_of_flight: Series[str] = pa.Field(nullable=True)
    Report_Status: Series[str] = pa.Field(nullable=True)
    Publication_Date: Series[pd.DatetimeTZDtype] = pa.Field(
        dtype_kwargs={"unit": "ns", "tz": "EST"}, nullable=True
    )

    class Config:
        """Input schema config"""

        coerce = True
