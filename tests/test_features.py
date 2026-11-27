"""Unit tests for meteorological feature engineering (src/features.py)."""

import math
import unittest

import numpy as np

from src.features import FEATURE_NAMES, build_single_feature_row


class TestFeatureEngineering(unittest.TestCase):
    def test_feature_columns_and_order(self) -> None:
        df = build_single_feature_row(
            max_temp=22.0,
            min_temp=12.0,
            wind_speed=3.5,
            cloud_cover=40.0,
            humidity=65.0,
            pressure=1013.25,
            uv_index=5.0,
            month=5,
        )
        self.assertEqual(list(df.columns), FEATURE_NAMES)
        self.assertEqual(df.shape, (1, 18))
        self.assertFalse(df.isna().any().any())
        self.assertTrue(np.isfinite(df.to_numpy()).all())

    def test_dew_point_at_full_saturation(self) -> None:
        df = build_single_feature_row(
            max_temp=20.0,
            min_temp=10.0,
            wind_speed=2.0,
            cloud_cover=100.0,
            humidity=100.0,
            pressure=1010.0,
            uv_index=1.0,
            month=11,
        )
        # At 100% humidity, dew point equals mean temperature (15.0 °C) and spread is 0
        self.assertAlmostEqual(float(df["Temp_Mean"].iloc[0]), 15.0, places=4)
        self.assertAlmostEqual(float(df["Dew_Point"].iloc[0]), 15.0, places=3)
        self.assertAlmostEqual(float(df["Dew_Spread"].iloc[0]), 0.0, places=3)

    def test_min_max_temperature_auto_swap_and_clamping(self) -> None:
        df = build_single_feature_row(
            max_temp=5.0,
            min_temp=18.0,  # deliberately inverted
            wind_speed=-4.0,
            cloud_cover=140.0,
            humidity=0.0,
            pressure=1005.0,
            uv_index=-1.0,
            month=15,
        )
        self.assertEqual(float(df["Max_Temperature"].iloc[0]), 18.0)
        self.assertEqual(float(df["Min_Temperature"].iloc[0]), 5.0)
        self.assertEqual(float(df["Temp_Diff"].iloc[0]), 13.0)
        self.assertEqual(float(df["WindSpeed"].iloc[0]), 0.0)
        self.assertEqual(float(df["CloudCover"].iloc[0]), 100.0)
        self.assertEqual(float(df["Humidity"].iloc[0]), 1.0)
        self.assertEqual(float(df["Uvi"].iloc[0]), 0.0)

    def test_cyclical_month_continuity(self) -> None:
        dec = build_single_feature_row(
            max_temp=10, min_temp=4, wind_speed=3, cloud_cover=50,
            humidity=70, pressure=1013, uv_index=2, month=12,
        )
        jan = build_single_feature_row(
            max_temp=10, min_temp=4, wind_speed=3, cloud_cover=50,
            humidity=70, pressure=1013, uv_index=2, month=1,
        )
        jul = build_single_feature_row(
            max_temp=10, min_temp=4, wind_speed=3, cloud_cover=50,
            humidity=70, pressure=1013, uv_index=2, month=7,
        )
        dist_dec_jan = math.hypot(
            float(dec["Month_Sin"].iloc[0] - jan["Month_Sin"].iloc[0]),
            float(dec["Month_Cos"].iloc[0] - jan["Month_Cos"].iloc[0]),
        )
        dist_dec_jul = math.hypot(
            float(dec["Month_Sin"].iloc[0] - jul["Month_Sin"].iloc[0]),
            float(dec["Month_Cos"].iloc[0] - jul["Month_Cos"].iloc[0]),
        )
        self.assertLess(dist_dec_jan, dist_dec_jul)


if __name__ == "__main__":
    unittest.main()
