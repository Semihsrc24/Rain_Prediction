"""Shared meteorological feature engineering for training and inference.

Derives physics-informed atmospheric features strictly from the 7 core weather
inputs collected in the UI plus the calendar month, ensuring 100% alignment
between model training and real-time inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_NAMES: list[str] = [
    "Max_Temperature",
    "Min_Temperature",
    "Temp_Mean",
    "Temp_Diff",
    "WindSpeed",
    "CloudCover",
    "Humidity",
    "Pressure",
    "Uvi",
    "Dew_Point",
    "Dew_Spread",
    "Pressure_Dev",
    "Cloud_Humidity",
    "Storm_Index",
    "Clear_Sky_Index",
    "Hum_Wind",
    "Month_Sin",
    "Month_Cos",
]


def engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute engineered meteorological features from raw weather columns.

    Required input columns:
        Max_Temperature, Min_Temperature, WindSpeed (m/s), CloudCover (%),
        Humidity (%), Pressure (hPa), Uvi, Month (1-12)
    """
    df = raw_df.copy()

    max_t = np.maximum(df["Max_Temperature"].astype(float), df["Min_Temperature"].astype(float))
    min_t = np.minimum(df["Max_Temperature"].astype(float), df["Min_Temperature"].astype(float))
    wind = np.maximum(df["WindSpeed"].astype(float), 0.0)
    cloud = np.clip(df["CloudCover"].astype(float), 0.0, 100.0)
    humidity = np.clip(df["Humidity"].astype(float), 1.0, 100.0)
    pressure = df["Pressure"].astype(float)
    uvi = np.maximum(df["Uvi"].astype(float), 0.0)
    month = np.clip(df["Month"].astype(float), 1.0, 12.0)

    temp_mean = (max_t + min_t) / 2.0
    temp_diff = max_t - min_t

    # Magnus-Tetens approximation for dew point (°C) and dew point depression
    gamma = np.log(humidity / 100.0) + (17.625 * temp_mean) / (243.04 + temp_mean)
    dew_point = (243.04 * gamma) / (17.625 - gamma)
    dew_spread = temp_mean - dew_point

    out = pd.DataFrame(
        {
            "Max_Temperature": max_t,
            "Min_Temperature": min_t,
            "Temp_Mean": temp_mean,
            "Temp_Diff": temp_diff,
            "WindSpeed": wind,
            "CloudCover": cloud,
            "Humidity": humidity,
            "Pressure": pressure,
            "Uvi": uvi,
            "Dew_Point": dew_point,
            "Dew_Spread": dew_spread,
            "Pressure_Dev": pressure - 1013.25,
            "Cloud_Humidity": (cloud * humidity) / 100.0,
            "Storm_Index": (cloud * np.maximum(0.0, 1016.0 - pressure)) / 100.0,
            "Clear_Sky_Index": uvi / (cloud + 5.0),
            "Hum_Wind": (humidity * wind) / 100.0,
            "Month_Sin": np.sin(2.0 * np.pi * month / 12.0),
            "Month_Cos": np.cos(2.0 * np.pi * month / 12.0),
        },
        index=df.index,
    )
    return out[FEATURE_NAMES]


def build_single_feature_row(
    *,
    max_temp: float,
    min_temp: float,
    wind_speed: float,
    cloud_cover: float,
    humidity: float,
    pressure: float,
    uv_index: float,
    month: int,
) -> pd.DataFrame:
    """Create a 1-row engineered feature DataFrame ready for model inference."""
    raw = pd.DataFrame(
        [
            {
                "Max_Temperature": float(max_temp),
                "Min_Temperature": float(min_temp),
                "WindSpeed": float(wind_speed),
                "CloudCover": float(cloud_cover),
                "Humidity": float(humidity),
                "Pressure": float(pressure),
                "Uvi": float(uv_index),
                "Month": int(month),
            }
        ]
    )
    return engineer_features(raw)
