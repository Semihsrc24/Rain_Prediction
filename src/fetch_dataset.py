"""Download 6 years (2020-2025) of daily historical weather across Turkey's 7 climate regions.

Uses the free Open-Meteo Historical Weather Archive API (no API key required) to
build data/turkey_weather_dataset.csv (~15,300 daily observations across İstanbul,
Ankara, İzmir, Antalya, Trabzon, Erzurum, and Diyarbakır).
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

CLIMATE_CITIES: dict[str, tuple[str, float, float]] = {
    "İstanbul": ("Marmara", 41.0082, 28.9784),
    "Ankara": ("Central Anatolia", 39.9334, 32.8597),
    "İzmir": ("Aegean", 38.4192, 27.1287),
    "Antalya": ("Mediterranean", 36.8969, 30.7133),
    "Trabzon": ("Black Sea", 41.0015, 39.7178),
    "Erzurum": ("Eastern Anatolia", 39.9086, 41.2769),
    "Diyarbakır": ("Southeastern Anatolia", 37.9144, 40.2306),
}

DAILY_FIELDS = ",".join(
    [
        "temperature_2m_max",
        "temperature_2m_min",
        "wind_speed_10m_mean",
        "cloud_cover_mean",
        "relative_humidity_2m_mean",
        "pressure_msl_mean",
        "shortwave_radiation_sum",
        "precipitation_sum",
    ]
)


def fetch_turkey_historical_dataset(
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    output_path: str | None = None,
) -> pd.DataFrame:
    """Fetch and save multi-region historical weather observations for Turkey."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if output_path is None:
        output_path = os.path.join(project_root, "data", "turkey_weather_dataset.csv")

    items = list(CLIMATE_CITIES.items())
    lats = ",".join(f"{lat:.4f}" for _, (_, lat, _) in items)
    lons = ",".join(f"{lon:.4f}" for _, (_, _, lon) in items)

    params = {
        "latitude": lats,
        "longitude": lons,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "Europe/Istanbul",
        "wind_speed_unit": "ms",
        "daily": DAILY_FIELDS,
    }

    payload = None
    for attempt in range(4):
        resp = requests.get(ARCHIVE_URL, params=params, timeout=45)
        if resp.status_code == 429:
            wait = 5 * (attempt + 1)
            print(f"Rate limited (429), retrying in {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        payload = resp.json()
        break

    if payload is None:
        raise RuntimeError("Failed to download dataset after retries due to HTTP 429.")
    if isinstance(payload, dict):
        payload = [payload]

    frames: list[pd.DataFrame] = []
    for (city, (region, _, _)), entry in zip(items, payload):
        daily = entry["daily"]
        sw_rad = pd.to_numeric(pd.Series(daily["shortwave_radiation_sum"]), errors="coerce")
        # Physical clear/cloudy-sky conversion from daily shortwave radiation (MJ/m²) to max UV index
        uv_est = np.clip((sw_rad / 3.15).round(2), 0.2, 11.5)

        city_df = pd.DataFrame(
            {
                "date": daily["time"],
                "city": city,
                "region": region,
                "Max_Temperature": daily["temperature_2m_max"],
                "Min_Temperature": daily["temperature_2m_min"],
                "WindSpeed": daily["wind_speed_10m_mean"],
                "CloudCover": np.clip(pd.to_numeric(daily["cloud_cover_mean"], errors="coerce"), 0, 100),
                "Humidity": np.clip(
                    pd.to_numeric(daily["relative_humidity_2m_mean"], errors="coerce"), 1, 100
                ),
                "Pressure": daily["pressure_msl_mean"],
                "Shortwave_Radiation": sw_rad,
                "Uvi": uv_est,
                "Precipitation": daily["precipitation_sum"],
            }
        )
        frames.append(city_df)
        print(f"Fetched {city:12s} ({region:21s}): {len(city_df)} days")

    combined = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"\nSaved {len(combined):,} total observations to {output_path}")
    return combined


if __name__ == "__main__":
    fetch_turkey_historical_dataset()
