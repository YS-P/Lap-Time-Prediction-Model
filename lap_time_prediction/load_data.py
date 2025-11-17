# load_data.py
import fastf1
import pandas as pd
from config import CACHE_DIR, YEAR, GRAND_PRIX, SESSION
import os

def load_session():
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)

    session = fastf1.get_session(YEAR, GRAND_PRIX, SESSION)
    session.load()

    laps = session.laps
    weather = session.weather_data

    laps = laps.sort_values("LapStartTime")
    weather = weather.sort_values("Time")

    merged = pd.merge_asof(
        laps,
        weather,
        left_on="LapStartTime",
        right_on="Time",
        direction="backward"
    )

    # Drop rows with NaT in LapTime
    merged = merged.dropna(subset=['LapTime'])

    # Remove columns that directly encode lap time (to avoid leakage)
    patterns = ['Sector', 'Speed', 'Pit', 'SessionTime',
                'StartTime', 'Time_x', 'Time_y', 'LapStartTime']
    drop_cols = [col for col in merged.columns if any(p in col for p in patterns)]
    merged = merged.drop(columns=drop_cols)

    return merged
