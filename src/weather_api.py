"""Turkish provinces catalog and Open-Meteo live weather API client (no API key required)."""

from __future__ import annotations

import datetime
from typing import Any

import requests

# All 81 provinces of Turkey with geographic coordinates (latitude, longitude)
TURKEY_CITIES: dict[str, tuple[float, float]] = {
    "Adana": (37.0017, 35.3289),
    "Adıyaman": (37.7648, 38.2764),
    "Afyonkarahisar": (38.7570, 30.5436),
    "Ağrı": (39.7191, 43.0517),
    "Aksaray": (38.3687, 34.0377),
    "Amasya": (40.6538, 35.8331),
    "Ankara": (39.9334, 32.8597),
    "Antalya": (36.8969, 30.7133),
    "Ardahan": (41.1104, 42.7022),
    "Artvin": (41.1828, 41.8186),
    "Aydın": (37.8560, 27.8458),
    "Balıkesir": (39.6484, 27.8826),
    "Bartın": (41.5811, 32.4617),
    "Batman": (37.8872, 41.1350),
    "Bayburt": (40.2556, 40.2247),
    "Bilecik": (40.1508, 29.9839),
    "Bingöl": (38.8847, 40.4919),
    "Bitlis": (38.3983, 42.1236),
    "Bolu": (40.7358, 31.6119),
    "Burdur": (37.7203, 30.2906),
    "Bursa": (40.1828, 29.0660),
    "Çanakkale": (40.1556, 26.4142),
    "Çankırı": (40.6017, 33.6139),
    "Çorum": (40.5492, 34.9536),
    "Denizli": (37.7767, 29.0869),
    "Diyarbakır": (37.9144, 40.2306),
    "Düzce": (40.8433, 31.1564),
    "Edirne": (41.6771, 26.5556),
    "Elazığ": (38.6744, 39.2228),
    "Erzincan": (39.7500, 39.5000),
    "Erzurum": (39.9086, 41.2769),
    "Eskişehir": (39.7767, 30.5206),
    "Gaziantep": (37.0662, 37.3833),
    "Giresun": (40.9122, 38.3894),
    "Gümüşhane": (40.4592, 39.4811),
    "Hakkari": (37.5747, 43.7408),
    "Hatay": (36.2028, 36.1603),
    "Iğdır": (39.9167, 44.0500),
    "Isparta": (37.7644, 30.5567),
    "İstanbul": (41.0082, 28.9784),
    "İzmir": (38.4192, 27.1287),
    "Kahramanmaraş": (37.5750, 36.9372),
    "Karabük": (41.2044, 32.6217),
    "Karaman": (37.1867, 33.2156),
    "Kars": (40.6000, 43.1000),
    "Kastamonu": (41.3786, 33.7833),
    "Kayseri": (38.7312, 35.4787),
    "Kırıkkale": (39.8461, 33.5156),
    "Kırklareli": (41.7353, 27.2250),
    "Kırşehir": (39.1425, 34.1611),
    "Kilis": (36.7186, 37.1122),
    "Kocaeli": (40.7667, 29.9167),
    "Konya": (37.8746, 32.4932),
    "Kütahya": (39.4242, 29.9839),
    "Malatya": (38.3556, 38.3094),
    "Manisa": (38.6192, 27.4286),
    "Mardin": (37.3122, 40.7350),
    "Mersin": (36.7997, 34.6189),
    "Muğla": (37.2153, 28.3636),
    "Muş": (38.7436, 41.5067),
    "Nevşehir": (38.6242, 34.7142),
    "Niğde": (37.9664, 34.6803),
    "Ordu": (40.9850, 37.8786),
    "Osmaniye": (37.0742, 36.2536),
    "Rize": (41.0206, 40.5231),
    "Sakarya": (40.7786, 30.4033),
    "Samsun": (41.2867, 36.3300),
    "Şanlıurfa": (37.1672, 38.7950),
    "Siirt": (37.9350, 41.9325),
    "Sinop": (42.0233, 35.1536),
    "Sivas": (39.7475, 37.0172),
    "Şırnak": (37.5164, 42.4611),
    "Tekirdağ": (40.9833, 27.5167),
    "Tokat": (40.3167, 36.5500),
    "Trabzon": (41.0015, 39.7178),
    "Tunceli": (39.1094, 39.5494),
    "Uşak": (38.6828, 29.4081),
    "Van": (38.4942, 43.3822),
    "Yalova": (40.6506, 29.2767),
    "Yozgat": (39.8189, 34.8147),
    "Zonguldak": (41.4553, 31.7894),
}

# WMO weather interpretation codes -> (description, emoji, is_rainy)
WMO_CODES: dict[int, tuple[str, str, bool]] = {
    0: ("Clear Sky", "☀️", False),
    1: ("Mainly Clear", "🌤️", False),
    2: ("Partly Cloudy", "⛅", False),
    3: ("Overcast", "☁️", False),
    45: ("Fog", "🌫️", False),
    48: ("Depositing Rime Fog", "🌫️", False),
    51: ("Light Drizzle", "🌦️", True),
    53: ("Moderate Drizzle", "🌦️", True),
    55: ("Dense Drizzle", "🌧️", True),
    56: ("Freezing Drizzle", "🌧️", True),
    57: ("Dense Freezing Drizzle", "🌧️", True),
    61: ("Slight Rain", "🌦️", True),
    63: ("Moderate Rain", "🌧️", True),
    65: ("Heavy Rain", "🌧️", True),
    66: ("Freezing Rain", "🌧️", True),
    67: ("Heavy Freezing Rain", "🌧️", True),
    71: ("Slight Snowfall", "🌨️", True),
    73: ("Moderate Snowfall", "🌨️", True),
    75: ("Heavy Snowfall", "❄️", True),
    77: ("Snow Grains", "🌨️", True),
    80: ("Slight Rain Showers", "🌦️", True),
    81: ("Moderate Rain Showers", "🌧️", True),
    82: ("Violent Rain Showers", "⛈️", True),
    85: ("Slight Snow Showers", "🌨️", True),
    86: ("Heavy Snow Showers", "❄️", True),
    95: ("Thunderstorm", "⛈️", True),
    96: ("Thunderstorm w/ Hail", "⛈️", True),
    99: ("Heavy Thunderstorm w/ Hail", "⛈️", True),
}

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

_TURKISH_ASCII_MAP = str.maketrans("çğıİöşüÇĞÖŞÜ", "cgiiosucgosu")


def _normalize_city_key(name: str) -> str:
    return name.strip().translate(_TURKISH_ASCII_MAP).lower()


_CITY_LOOKUP: dict[str, str] = {
    _normalize_city_key(city): city for city in TURKEY_CITIES
}


def resolve_city_name(city_name: str, default: str = "İstanbul") -> str:
    """Resolve user input (including ASCII/lowercase variants) to canonical Turkish province name."""
    if not city_name:
        return default
    if city_name in TURKEY_CITIES:
        return city_name
    return _CITY_LOOKUP.get(_normalize_city_key(city_name), default)


def decode_wmo(code: int | None) -> tuple[str, str, bool]:
    if code is None:
        return ("Unknown", "🌡️", False)
    return WMO_CODES.get(int(code), ("Cloudy", "☁️", False))


def fetch_city_forecast(city_name: str, timeout: int = 12) -> dict[str, Any]:
    """Fetch live current weather and 7-day forecast for a Turkish city via Open-Meteo."""
    canonical_city = resolve_city_name(city_name)
    lat, lon = TURKEY_CITIES[canonical_city]

    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "Europe/Istanbul",
        "forecast_days": 7,
        "wind_speed_unit": "ms",
        "current": (
            "temperature_2m,relative_humidity_2m,apparent_temperature,"
            "precipitation,weather_code,cloud_cover,pressure_msl,wind_speed_10m"
        ),
        "daily": (
            "weather_code,temperature_2m_max,temperature_2m_min,"
            "precipitation_sum,precipitation_probability_max,wind_speed_10m_max,uv_index_max"
        ),
        "hourly": "cloud_cover,relative_humidity_2m,pressure_msl,wind_speed_10m",
    }

    resp = requests.get(OPEN_METEO_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()

    current = payload.get("current", {})
    daily = payload.get("daily", {})
    hourly = payload.get("hourly", {})

    # Extract tomorrow (index 1, fallback to index 0)
    dates = daily.get("time", [])
    idx = 1 if len(dates) > 1 else 0
    target_date = dates[idx] if dates else datetime.date.today().isoformat()

    # Compute true daily means for tomorrow from hourly data so features align with Bdataset.csv
    hourly_times = hourly.get("time", [])
    hour_indices = [i for i, t in enumerate(hourly_times) if t.startswith(target_date)]

    def mean_for_hours(key: str, fallback: float) -> float:
        series = hourly.get(key, [])
        vals = [series[i] for i in hour_indices if i < len(series) and series[i] is not None]
        return round(sum(vals) / len(vals), 2) if vals else fallback

    tomorrow_code = daily.get("weather_code", [0])[idx]
    tomorrow_desc, tomorrow_emoji, tomorrow_is_rainy = decode_wmo(tomorrow_code)

    tomorrow_summary = {
        "date": target_date,
        "month": int(target_date.split("-")[1]),
        "max_temp": round(float(daily["temperature_2m_max"][idx]), 1),
        "min_temp": round(float(daily["temperature_2m_min"][idx]), 1),
        "wind_speed": mean_for_hours(
            "wind_speed_10m", round(float(daily["wind_speed_10m_max"][idx]) * 0.65, 2)
        ),
        "wind_max": round(float(daily["wind_speed_10m_max"][idx]), 1),
        "cloud_cover": round(mean_for_hours("cloud_cover", 50.0), 1),
        "humidity": round(mean_for_hours("relative_humidity_2m", 65.0), 1),
        "pressure": round(mean_for_hours("pressure_msl", 1013.0), 1),
        "uv_index": round(float(daily["uv_index_max"][idx] or 0.0), 1),
        "precip_mm": round(float(daily["precipitation_sum"][idx] or 0.0), 1),
        "precip_prob": int(daily.get("precipitation_probability_max", [0])[idx] or 0),
        "condition": tomorrow_desc,
        "emoji": tomorrow_emoji,
        "is_rainy": tomorrow_is_rainy,
    }

    curr_desc, curr_emoji, _ = decode_wmo(current.get("weather_code"))
    current_summary = {
        "temp": round(float(current.get("temperature_2m", 0.0)), 1),
        "feels_like": round(float(current.get("apparent_temperature", 0.0)), 1),
        "humidity": int(current.get("relative_humidity_2m", 0)),
        "wind_speed": round(float(current.get("wind_speed_10m", 0.0)), 1),
        "cloud_cover": int(current.get("cloud_cover", 0)),
        "pressure": round(float(current.get("pressure_msl", 1013.0)), 1),
        "precip_mm": round(float(current.get("precipitation", 0.0)), 1),
        "condition": curr_desc,
        "emoji": curr_emoji,
    }

    week_forecast = []
    for i, d_str in enumerate(dates[:5]):
        w_desc, w_emoji, _ = decode_wmo(daily["weather_code"][i])
        day_label = datetime.date.fromisoformat(d_str).strftime("%a %d")
        week_forecast.append(
            {
                "date": d_str,
                "label": day_label,
                "max_temp": round(float(daily["temperature_2m_max"][i]), 1),
                "min_temp": round(float(daily["temperature_2m_min"][i]), 1),
                "precip_mm": round(float(daily["precipitation_sum"][i] or 0.0), 1),
                "precip_prob": int(daily.get("precipitation_probability_max", [0] * len(dates))[i] or 0),
                "condition": w_desc,
                "emoji": w_emoji,
            }
        )

    return {
        "city": canonical_city,
        "lat": lat,
        "lon": lon,
        "current": current_summary,
        "tomorrow": tomorrow_summary,
        "week": week_forecast,
        "updated_at": datetime.datetime.now().strftime("%H:%M"),
    }


def fetch_turkey_map_weather(timeout: int = 15) -> list[dict[str, Any]]:
    """Fetch real-time weather for all 81 Turkish provinces in 2 fast batch requests."""
    items = list(TURKEY_CITIES.items())
    chunk_size = 41
    results: list[dict[str, Any]] = []

    for start in range(0, len(items), chunk_size):
        chunk = items[start : start + chunk_size]
        lats = ",".join(f"{lat:.4f}" for _, (lat, _) in chunk)
        lons = ",".join(f"{lon:.4f}" for _, (_, lon) in chunk)

        params = {
            "latitude": lats,
            "longitude": lons,
            "timezone": "Europe/Istanbul",
            "wind_speed_unit": "ms",
            "current": (
                "temperature_2m,relative_humidity_2m,apparent_temperature,"
                "precipitation,weather_code,cloud_cover,pressure_msl,wind_speed_10m"
            ),
        }

        resp = requests.get(OPEN_METEO_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict):
            payload = [payload]

        for (city, (lat, lon)), entry in zip(chunk, payload):
            curr = entry.get("current", {})
            code = curr.get("weather_code", 0)
            desc, emoji, is_rainy = decode_wmo(code)
            precip = round(float(curr.get("precipitation", 0.0) or 0.0), 1)
            results.append(
                {
                    "city": city,
                    "lat": lat,
                    "lon": lon,
                    "temp": round(float(curr.get("temperature_2m", 0.0)), 1),
                    "feels_like": round(float(curr.get("apparent_temperature", 0.0)), 1),
                    "humidity": int(curr.get("relative_humidity_2m", 0)),
                    "wind": round(float(curr.get("wind_speed_10m", 0.0)), 1),
                    "cloud": int(curr.get("cloud_cover", 0)),
                    "pressure": round(float(curr.get("pressure_msl", 1013.0)), 1),
                    "precip": precip,
                    "desc": desc,
                    "emoji": emoji,
                    "is_rainy": bool(is_rainy or precip > 0.1),
                }
            )

    return results
