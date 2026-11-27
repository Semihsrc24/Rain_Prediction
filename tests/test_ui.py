"""Functional PyQt6 UI tests verifying every widget and handler in src/ui.py."""

import os
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")

from PyQt6.QtWidgets import QApplication, QDialogButtonBox

from src.ui import EmailAlertDialog, WeatherApp


class TestWeatherAppUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self) -> None:
        # Prevent automatic live HTTP calls during unit test startup
        with patch.object(WeatherApp, "fetch_selected_city_forecast", return_value=None):
            self.win = WeatherApp()

    def tearDown(self) -> None:
        self.win.close()

    def test_initial_state_and_model_loaded(self) -> None:
        self.assertIsNotNone(self.win.model)
        self.assertEqual(self.win.city_combo.count(), 81)
        self.assertEqual(self.win.city_combo.currentText(), "İstanbul")
        self.assertIn("Ensemble", self.win.model_meta_lbl.text())
        self.assertIn("15,344", self.win.model_meta_lbl.text())

    def test_presets_update_verdict_badge_and_bar(self) -> None:
        # Clear Day
        self.win._apply_preset((27.5, 16.0, 2.8, 12.0, 44.0, 1021.0, 7.2))
        self.assertEqual(self.win.verdict_label.text(), "No Rain Expected")
        self.assertEqual(self.win.weather_badge.state, "sunny")
        self.assertLess(self.win.prob_bar.value(), 25)

        # Storm Front
        self.win._apply_preset((12.0, 7.5, 7.6, 94.0, 88.0, 1001.5, 1.4))
        self.assertEqual(self.win.verdict_label.text(), "Rain Expected")
        self.assertEqual(self.win.weather_badge.state, "rainy")
        self.assertGreater(self.win.prob_bar.value(), 70)

    def test_inverted_min_max_temperature_handled_safely(self) -> None:
        self.win.spinboxes["Max_Temperature"].setValue(8.0)
        self.win.spinboxes["Min_Temperature"].setValue(20.0)
        self.win.predict_manual()
        self.assertEqual(self.win.spinboxes["Max_Temperature"].value(), 20.0)
        self.assertEqual(self.win.spinboxes["Min_Temperature"].value(), 8.0)
        self.assertTrue(self.win.prob_big_label.text().endswith("%"))

    def test_forecast_loaded_populates_tiles_and_autofills(self) -> None:
        sample = {
            "city": "Ankara",
            "lat": 39.9334,
            "lon": 32.8597,
            "updated_at": "12:00",
            "current": {
                "temp": 14.0,
                "feels_like": 13.0,
                "humidity": 60,
                "wind_speed": 3.0,
                "cloud_cover": 40,
                "pressure": 1014.0,
                "precip_mm": 0.0,
                "condition": "Partly Cloudy",
                "emoji": "⛅",
            },
            "tomorrow": {
                "date": "2026-11-28",
                "month": 11,
                "max_temp": 11.5,
                "min_temp": 4.0,
                "wind_speed": 6.2,
                "wind_max": 9.0,
                "cloud_cover": 92.0,
                "humidity": 86.0,
                "pressure": 1002.0,
                "uv_index": 1.6,
                "precip_mm": 6.4,
                "precip_prob": 85,
                "condition": "Moderate Rain",
                "emoji": "🌧️",
                "is_rainy": True,
            },
            "week": [
                {
                    "date": f"2026-11-{27 + i:02d}",
                    "label": f"Day {i + 1}",
                    "max_temp": 12.0 + i,
                    "min_temp": 5.0 + i,
                    "precip_mm": 0.0,
                    "precip_prob": 20 * i,
                    "condition": "Cloudy",
                    "emoji": "☁️",
                }
                for i in range(5)
            ],
        }
        self.win._expected_city = "Ankara"
        self.win._handle_forecast_result(sample, auto_fill=True, req_city="Ankara")

        self.assertIn("Ankara", self.win.fc_city_title.text())
        self.assertEqual(self.win.metric_labels["temp_range"].text(), "4.0° / 11.5°C")
        self.assertEqual(self.win.metric_labels["precip"].text(), "6.4 mm (85%)")
        self.assertEqual(self.win.spinboxes["CloudCover"].value(), 92.0)
        self.assertEqual(self.win.verdict_label.text(), "Rain Expected")

        # Stale response for a different city should be ignored
        stale = dict(sample, city="İzmir")
        self.win._handle_forecast_result(stale, auto_fill=False, req_city="İzmir")
        self.assertIn("Ankara", self.win.fc_city_title.text())

    def test_offline_forecast_error_graceful(self) -> None:
        self.win._expected_city = "İstanbul"
        self.win._handle_forecast_error("DNS failure", req_city="İstanbul")
        self.assertTrue(self.win.sync_btn.isEnabled())
        self.assertIn("Offline mode", self.win.fc_current_sub.text())

    def test_page_navigation_and_map_controls(self) -> None:
        with patch.object(self.win.map_page, "ensure_loaded", return_value=None):
            self.win._switch_page(1)
            self.assertEqual(self.win.stack.currentIndex(), 1)
            self.assertTrue(self.win.nav_map_btn.property("active"))
            self.assertFalse(self.win.nav_pred_btn.property("active"))

        mock_cities = [
            {
                "city": "Antalya", "lat": 36.89, "lon": 30.71, "temp": 22.4,
                "feels_like": 22.0, "humidity": 55, "wind": 3.0, "cloud": 10,
                "pressure": 1018.0, "precip": 0.0, "desc": "Clear Sky",
                "emoji": "☀️", "is_rainy": False,
            },
            {
                "city": "Ardahan", "lat": 41.11, "lon": 42.70, "temp": -3.2,
                "feels_like": -6.0, "humidity": 88, "wind": 5.5, "cloud": 95,
                "pressure": 1004.0, "precip": 2.4, "desc": "Moderate Snowfall",
                "emoji": "🌨️", "is_rainy": True,
            },
        ]
        self.win.map_page._on_map_loaded(mock_cities)
        self.assertIn("Antalya", self.win.map_page.stat_warmest.findChild(type(self.win.verdict_label), "pill_value").text())
        self.assertIn("Ardahan", self.win.map_page.stat_coldest.findChild(type(self.win.verdict_label), "pill_value").text())
        self.assertEqual(self.win.map_page.stat_rainy.findChild(type(self.win.verdict_label), "pill_value").text(), "1 / 2")

        # Filter & theme toggles
        self.win.map_page._on_filter_changed(1)
        self.assertEqual(self.win.map_page.filter_mode, "rainy")
        self.win.map_page._toggle_map_theme()
        self.assertFalse(self.win.map_page.dark_tiles)
        self.assertIn("Dark Map", self.win.map_page.theme_btn.text())

    def test_email_dialog_disables_send_without_credentials(self) -> None:
        with patch.dict(os.environ, {"SMTP_USER": "", "SMTP_PASSWORD": ""}, clear=False):
            dlg = EmailAlertDialog("Test Subject", "Test Body", parent=self.win)
            buttons = dlg.findChild(QDialogButtonBox)
            send_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
            self.assertFalse(send_btn.isEnabled())
            dlg.close()

        with patch.dict(os.environ, {"SMTP_USER": "demo@example.com", "SMTP_PASSWORD": "secret"}, clear=False):
            dlg = EmailAlertDialog("Test Subject", "Test Body", parent=self.win)
            buttons = dlg.findChild(QDialogButtonBox)
            send_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
            self.assertTrue(send_btn.isEnabled())
            dlg.close()


if __name__ == "__main__":
    unittest.main()
