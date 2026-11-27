# 📊 Exploratory Data Analysis (EDA) & Data Quality Report

![EDA Overview](eda_overview.png)

## 1. Dataset Overview

- **Primary Dataset:** `data/turkey_weather_dataset.csv`
- **Records:** `15,344` daily meteorological observations
- **Geographic Coverage:** `7` cities representing all 7 climate regions of Turkey
- **Time Span:** `2020-01-01` → `2025-12-31` (6 full years)
- **Target Definition:** Binary rain classification (`Precipitation > 0.2 mm` → `1`, else `0`)
- **Class Balance:** `10,180` No Rain (`66.35%`) vs `5,164` Rain (`33.65%`)

## 2. Data Quality & Integrity Audit

| Check | Result | Status |
| :--- | :--- | :---: |
| Missing values (core features) | `0` | ✅ Clean |
| Exact duplicate rows | `0` | ✅ Clean |
| Duplicate `(city, date)` keys | `0` | ✅ Clean |
| Inverted temperatures (`Min > Max`) | `0` | ✅ Clean |
| Out-of-range Cloud Cover (`<0` or `>100%`) | `0` | ✅ Clean |
| Out-of-range Humidity (`<0` or `>100%`) | `0` | ✅ Clean |
| Extreme Pressure (`<940` or `>1060 hPa`) | `0` | ✅ Clean |

### Legacy Dataset Audit (`data/weather_dataset.csv`)

The original dataset contained only `1,536` single-city rows (`2021-03-24` → `2025-06-06`), used **surface pressure** (which varies by ~110 hPa per 1,000m elevation and does not generalize to high-altitude cities like Ankara or Erzurum), and had minor reanalysis boundary anomalies:

- `cloud_cover_max (%)`: 6 out-of-[0,100] values (min=0, max=101)
- `cloud_cover_min (%)`: 4 out-of-[0,100] values (min=-1, max=100)

Switching to `data/turkey_weather_dataset.csv` expanded sample size by **10×** and replaced raw surface pressure with **Mean Sea-Level Pressure (`pressure_msl`)** for elevation-invariant forecasting across all 81 provinces.

## 3. Descriptive Statistics & IQR Outliers

| Feature | Min | 25% | Median | Mean | 75% | Max | Std | IQR Outliers |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Max_Temperature` | -13.6 | 13.5 | 20.3 | 20.6 | 27.9 | 45.4 | 9.99 | 14 (0.09%) |
| `Min_Temperature` | -26.3 | 4.6 | 10.6 | 10.4 | 17.1 | 31.1 | 8.75 | 83 (0.54%) |
| `WindSpeed` | 0.41 | 1.63 | 2.16 | 2.49 | 3.0 | 11.53 | 1.25 | 795 (5.18%) |
| `CloudCover` | 0.0 | 8.0 | 35.0 | 40.9 | 71.0 | 100.0 | 33.69 | 0 (0.0%) |
| `Humidity` | 9.0 | 51.0 | 66.0 | 63.23 | 78.0 | 98.0 | 18.5 | 3 (0.02%) |
| `Pressure` | 993.8 | 1010.4 | 1014.7 | 1015.12 | 1019.6 | 1038.8 | 6.68 | 60 (0.39%) |
| `Uvi` | 0.2 | 3.2 | 5.25 | 5.35 | 7.62 | 10.2 | 2.5 | 0 (0.0%) |
| `Precipitation` | 0.0 | 0.0 | 0.0 | 1.89 | 1.0 | 79.4 | 5.17 | 2656 (17.31%) |

> **Note on IQR Outliers:** Meteorological extremes (e.g., storm gusts, deep low-pressure fronts, and heavy precipitation spikes) are genuine weather events rather than sensor errors, so they are intentionally retained for training.

## 4. Regional Climate Breakdown (7 Regions of Turkey)

| City | Region | Days | Rainy Days (%) | Avg Max Temp (°C) | Avg Humidity (%) | Avg Cloud (%) | Annual Precip (mm) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trabzon | Black Sea | 2,192 | **53.5%** | 18.9 | 76.2 | 62.3 | 1,045 |
| Erzurum | Eastern Anatolia | 2,192 | **41.2%** | 13.6 | 61.6 | 42.1 | 656 |
| İstanbul | Marmara | 2,192 | **38.7%** | 19.4 | 75.0 | 48.2 | 744 |
| Ankara | Central Anatolia | 2,192 | **30.3%** | 19.0 | 59.4 | 40.5 | 406 |
| Antalya | Mediterranean | 2,192 | **24.9%** | 25.3 | 58.5 | 29.5 | 819 |
| İzmir | Aegean | 2,192 | **23.7%** | 24.1 | 60.8 | 31.2 | 648 |
| Diyarbakır | Southeastern Anatolia | 2,192 | **23.3%** | 23.8 | 51.2 | 32.5 | 508 |

## 5. Top Engineered Features Correlated with Rain

| Rank | Feature | Pearson Correlation (`r`) | Meteorological Interpretation |
| :---: | :--- | ---: | :--- |
| 1 | `Cloud_Humidity` | `+0.6425` | Combined moisture + cloud density (strongest positive predictor) |
| 2 | `CloudCover` | `+0.6058` | Direct daily mean cloud coverage |
| 3 | `Humidity` | `+0.5004` | Ambient moisture availability |
| 4 | `Temp_Diff` | `-0.4934` | Narrow diurnal temperature swing occurs on overcast rainy days |
| 5 | `Dew_Spread` | `-0.4459` | Smaller gap between temperature and dew point → air near saturation |
| 6 | `Clear_Sky_Index` | `-0.4256` | High UV relative to cloud cover strongly indicates dry weather |
| 7 | `Uvi` | `-0.3761` | Cloud attenuation suppresses surface UV index |
| 8 | `Storm_Index` | `+0.3733` | Cloud cover weighted by low-pressure depth below 1016 hPa |
| 9 | `Max_Temperature` | `-0.2987` | Cooler daytime highs accompany frontal rain systems |
| 10 | `Hum_Wind` | `+0.2589` | Moisture advection driven by wind |
| 11 | `Temp_Mean` | `-0.2219` | Derived atmospheric feature |
| 12 | `Month_Sin` | `+0.1702` | Derived atmospheric feature |
