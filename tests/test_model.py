"""Model integrity, accuracy regression gates, and meteorological sanity tests."""

import os
import unittest

import joblib
import numpy as np
import pandas as pd

from src.features import FEATURE_NAMES, build_single_feature_row
from src.train_model import find_best_threshold, load_dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "rain_model.pkl")
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "turkey_weather_dataset.csv")


class TestRainModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = joblib.load(MODEL_PATH)
        cls.model = cls.bundle["model"]

    def test_dataset_loading(self) -> None:
        X, y, years = load_dataset(DATASET_PATH)
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(X), len(years))
        self.assertGreaterEqual(len(X), 15000)
        self.assertEqual(list(X.columns), FEATURE_NAMES)
        self.assertEqual(set(y.unique()), {0, 1})
        self.assertFalse(X.isna().any().any())

    def test_model_bundle_schema_and_metrics(self) -> None:
        for key in ("model", "model_name", "feature_names", "threshold", "metrics", "feature_importances"):
            self.assertIn(key, self.bundle)

        self.assertEqual(self.bundle["feature_names"], FEATURE_NAMES)
        metrics = self.bundle["metrics"]
        self.assertGreaterEqual(metrics["accuracy"], 0.85)
        self.assertGreaterEqual(metrics["roc_auc"], 0.92)
        self.assertGreaterEqual(metrics["f1_rain"], 0.75)
        self.assertLessEqual(metrics["brier_score"], 0.11)
        self.assertGreaterEqual(metrics["temporal_2025_accuracy"], 0.85)
        self.assertGreaterEqual(metrics["temporal_2025_roc_auc"], 0.92)
        self.assertTrue(0.35 <= self.bundle["threshold"] <= 0.65)

    def test_meteorological_sanity_predictions(self) -> None:
        clear_day = build_single_feature_row(
            max_temp=30.0, min_temp=19.0, wind_speed=2.2, cloud_cover=5.0,
            humidity=38.0, pressure=1022.0, uv_index=8.0, month=7,
        )
        storm_day = build_single_feature_row(
            max_temp=11.0, min_temp=6.5, wind_speed=8.0, cloud_cover=96.0,
            humidity=90.0, pressure=998.0, uv_index=1.0, month=11,
        )

        clear_prob = float(self.model.predict_proba(clear_day)[0][1])
        storm_prob = float(self.model.predict_proba(storm_day)[0][1])

        self.assertTrue(0.0 <= clear_prob <= 1.0)
        self.assertTrue(0.0 <= storm_prob <= 1.0)
        self.assertLess(clear_prob, 0.20, f"Clear day rain prob too high: {clear_prob:.3f}")
        self.assertGreater(storm_prob, 0.75, f"Storm day rain prob too low: {storm_prob:.3f}")

    def test_rain_probability_increases_with_wet_low_pressure_conditions(self) -> None:
        dry_high = build_single_feature_row(
            max_temp=20.0, min_temp=12.0, wind_speed=3.0, cloud_cover=20.0,
            humidity=45.0, pressure=1022.0, uv_index=6.0, month=4,
        )
        wet_low = build_single_feature_row(
            max_temp=15.0, min_temp=10.0, wind_speed=6.5, cloud_cover=92.0,
            humidity=86.0, pressure=1002.0, uv_index=1.8, month=4,
        )
        p_dry = float(self.model.predict_proba(dry_high)[0][1])
        p_wet = float(self.model.predict_proba(wet_low)[0][1])
        self.assertGreater(p_wet, p_dry + 0.40)

    def test_threshold_optimizer(self) -> None:
        y_true = pd.Series([0, 0, 0, 0, 1, 1, 1])
        probs = np.array([0.1, 0.2, 0.3, 0.38, 0.45, 0.7, 0.9])
        thresh = find_best_threshold(y_true, probs)
        self.assertTrue(0.38 <= thresh <= 0.45)


if __name__ == "__main__":
    unittest.main()
