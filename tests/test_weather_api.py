"""Unit tests for Turkish provinces catalog and Open-Meteo API client (src/weather_api.py)."""

import unittest
from unittest.mock import MagicMock, patch

from src.weather_api import (
    TURKEY_CITIES,
    decode_wmo,
    fetch_city_forecast,
    fetch_turkey_map_weather,
    resolve_city_name,
)


class TestWeatherAPI(unittest.TestCase):
    def test_all_81_provinces_valid_coordinates(self) -> None:
        self.assertEqual(len(TURKEY_CITIES), 81)
        for city, (lat, lon) in TURKEY_CITIES.items():
            self.assertTrue(35.5 <= lat <= 42.5, f"Invalid latitude for {city}: {lat}")
            self.assertTrue(25.5 <= lon <= 45.0, f"Invalid longitude for {city}: {lon}")

    def test_resolve_city_name_turkish_and_ascii(self) -> None:
        self.assertEqual(resolve_city_name("İstanbul"), "İstanbul")
        self.assertEqual(resolve_city_name("istanbul"), "İstanbul")
        self.assertEqual(resolve_city_name("IZMIR"), "İzmir")
        self.assertEqual(resolve_city_name("sanliurfa"), "Şanlıurfa")
        self.assertEqual(resolve_city_name("canakkale"), "Çanakkale")
        self.assertEqual(resolve_city_name("mugla"), "Muğla")
        self.assertEqual(resolve_city_name("  ankara  "), "Ankara")
        self.assertEqual(resolve_city_name("NonExistentCity", default="İstanbul"), "İstanbul")
        self.assertEqual(resolve_city_name("", default=""), "")

    def test_decode_wmo_codes(self) -> None:
        self.assertEqual(decode_wmo(0), ("Clear Sky", "☀️", False))
        self.assertEqual(decode_wmo(63), ("Moderate Rain", "🌧️", True))
        self.assertEqual(decode_wmo(95), ("Thunderstorm", "⛈️", True))
        self.assertEqual(decode_wmo(None), ("Unknown", "🌡️", False))
        self.assertEqual(decode_wmo(999), ("Cloudy", "☁️", False))

    @patch("src.weather_api.requests.get")
    def test_fetch_city_forecast_parses_hourly_means(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "current": {
                "temperature_2m": 16.4,
                "apparent_temperature": 15.8,
                "relative_humidity_2m": 64,
                "wind_speed_10m": 4.2,
                "cloud_cover": 40,
                "pressure_msl": 1015.2,
                "precipitation": 0.0,
                "weather_code": 2,
            },
            "daily": {
                "time": ["2026-11-27", "2026-11-28", "2026-11-29", "2026-11-30", "2026-12-01"],
                "weather_code": [2, 61, 63, 3, 0],
                "temperature_2m_max": [18.0, 15.5, 13.0, 12.0, 14.0],
                "temperature_2m_min": [11.0, 9.5, 8.0, 6.0, 7.0],
                "precipitation_sum": [0.0, 4.6, 8.2, 0.2, 0.0],
                "precipitation_probability_max": [10, 80, 90, 25, 5],
                "wind_speed_10m_max": [5.0, 7.0, 6.0, 4.0, 3.0],
                "uv_index_max": [3.5, 2.2, 1.8, 3.0, 4.1],
            },
            "hourly": {
                "time": ["2026-11-28T06:00", "2026-11-28T12:00", "2026-11-28T18:00"],
                "cloud_cover": [80, 90, 100],
                "relative_humidity_2m": [75, 85, 80],
                "pressure_msl": [1008.0, 1006.0, 1007.0],
                "wind_speed_10m": [4.0, 6.0, 5.0],
            },
        }
        mock_get.return_value = mock_resp

        result = fetch_city_forecast("izmir")
        self.assertEqual(result["city"], "İzmir")
        self.assertEqual(result["tomorrow"]["date"], "2026-11-28")
        self.assertEqual(result["tomorrow"]["month"], 11)
        self.assertEqual(result["tomorrow"]["cloud_cover"], 90.0)
        self.assertEqual(result["tomorrow"]["humidity"], 80.0)
        self.assertEqual(result["tomorrow"]["pressure"], 1007.0)
        self.assertEqual(result["tomorrow"]["wind_speed"], 5.0)
        self.assertEqual(result["tomorrow"]["precip_mm"], 4.6)
        self.assertTrue(result["tomorrow"]["is_rainy"])
        self.assertEqual(len(result["week"]), 5)

    @patch("src.weather_api.requests.get")
    def test_fetch_turkey_map_weather_batches_81_cities(self, mock_get: MagicMock) -> None:
        def side_effect(url, params=None, timeout=15):
            count = len(params["latitude"].split(","))
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            resp.json.return_value = [
                {
                    "current": {
                        "temperature_2m": 12.0 + (i % 10),
                        "apparent_temperature": 11.0 + (i % 10),
                        "relative_humidity_2m": 70,
                        "wind_speed_10m": 3.5,
                        "cloud_cover": 50,
                        "pressure_msl": 1013.0,
                        "precipitation": 1.2 if i == 0 else 0.0,
                        "weather_code": 61 if i == 0 else 1,
                    }
                }
                for i in range(count)
            ]
            return resp

        mock_get.side_effect = side_effect
        cities = fetch_turkey_map_weather()

        self.assertEqual(len(cities), 81)
        self.assertEqual(mock_get.call_count, 2)  # 41 + 40 in 2 fast requests
        self.assertTrue(cities[0]["is_rainy"])
        self.assertFalse(cities[1]["is_rainy"])


if __name__ == "__main__":
    unittest.main()
