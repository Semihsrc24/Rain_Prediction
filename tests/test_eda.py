"""Unit tests for the Exploratory Data Analysis (EDA) pipeline (src/eda.py)."""

import os
import unittest

from src.eda import analyze_turkey_dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TURKEY_DATA = os.path.join(PROJECT_ROOT, "data", "turkey_weather_dataset.csv")


class TestEDA(unittest.TestCase):
    def test_turkey_dataset_quality_and_stats(self) -> None:
        clean_df, summary = analyze_turkey_dataset(TURKEY_DATA)
        self.assertEqual(len(clean_df), 15344)
        self.assertEqual(summary["cities"], 7)
        self.assertEqual(summary["total_missing"], 0)
        self.assertEqual(summary["dup_city_date"], 0)
        self.assertEqual(summary["inv_temp"], 0)
        self.assertEqual(summary["bad_cloud"], 0)
        self.assertEqual(summary["bad_hum"], 0)
        self.assertGreater(summary["correlations"]["Cloud_Humidity"], 0.55)
        self.assertEqual(len(summary["monthly_rain"]), 12)

    def test_eda_artifacts_exist(self) -> None:
        self.assertTrue(os.path.exists(os.path.join(PROJECT_ROOT, "docs", "EDA_REPORT.md")))
        self.assertTrue(os.path.exists(os.path.join(PROJECT_ROOT, "docs", "eda_overview.png")))


if __name__ == "__main__":
    unittest.main()
