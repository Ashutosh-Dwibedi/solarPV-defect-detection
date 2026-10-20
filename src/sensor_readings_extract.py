from pathlib import Path
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
PROJECT_ROOT = SRC_DIR.parent if SRC_DIR.name in ["src", "notebooks"] else SRC_DIR

DATA_DIR = PROJECT_ROOT / "data"
LABELS_PATH = DATA_DIR / "labels.csv"
OUTPUT_PATH = DATA_DIR / "paired_telemetry.csv"

df = pd.read_csv(LABELS_PATH)
df.columns = df.columns.str.strip()

N = len(df)
d = df["defect_prob"].values
wafer = df["wafer_type"].str.lower().str.strip().values
is_mono = (wafer == "mono").astype(float)

np.random.seed(42)

# Rooftop Meteorological Boundary Conditions
irradiance = 450.0 + np.random.beta(a=3.5, b=2.2, size=N) * 550.0
ambient_temp = np.random.normal(21.5, 3.2, size=N)

# Rooftop mounting factor: trapped backsheet air adds ~0.035 °C per W/m²
t_rise_ambient = irradiance * 0.035

# Localized Joule heating (I²R): severe micro-cracks form constricted current paths
t_hotspot = 8.5 * d + 6.0 * (d ** 2)

temp_noise = np.random.normal(0, 0.45, size=N)
cell_temp = ambient_temp + t_rise_ambient + t_hotspot + temp_noise

# STC peak rating per cell: Monocrystalline (~9.1 Wp) vs Polycrystalline (~8.5 Wp)
p_stc = np.where(is_mono, 9.1, 8.5)

# PV temperature derating
thermal_derate = 1.0 - 0.0038 * (cell_temp - 25.0)

# Theoretical expected output under instantaneous sunshine & thermal stress
p_expected = p_stc * (irradiance / 1000.0) * thermal_derate

# Structural degradation: active optical area loss & micro-crack shunting
degradation = 1.0 - (0.11 * d + 0.15 * (d ** 2))

field_losses = np.random.uniform(0.93, 0.98, size=N)
power_noise = np.random.normal(0, 0.06, size=N)

specific_power = np.clip(p_expected * degradation * field_losses + power_noise, a_min=0.2, a_max=None)

# Performance Ratio (PR): Actual vs. Clear-sky Expected
pr = specific_power / (p_stc * (irradiance / 1000.0) * thermal_derate)
pr = np.clip(pr + np.random.normal(0, 0.015, size=N), a_min=0.1, a_max=1.0)

h_base = (irradiance / 1000.0) * 4.6
h_weather_history = np.random.normal(0, 0.85, size=N)
h_cumulative = np.clip(h_base + h_weather_history, a_min=1.2, a_max=6.2)

# Specific Yield (kWh/kWp) incorporates historical generation + localized defect derating
yield_noise = np.random.normal(0, 0.08, size=N)
specific_yield = np.clip(h_cumulative * pr + yield_noise, a_min=0.2, a_max=None)

df["irradiance_w_m2"] = np.round(irradiance, 1)
df["cell_temp_c"] = np.round(cell_temp, 2)
df["specific_power_w"] = np.round(specific_power, 3)
df["specific_yield_kwh_kwp"] = np.round(specific_yield, 3)

df.to_csv(OUTPUT_PATH, index=False)

print(f"Generated normalized rooftop telemetry for {N} cells.")
print(f"Saved to: {OUTPUT_PATH}\n")

# Verify Pearson correlation matrix across numerical features
telemetry_cols = [
    "irradiance_w_m2", 
    "cell_temp_c", 
    "specific_power_w", 
    "specific_yield_kwh_kwp", 
    "defect_prob"
]
print("Pearson Correlation Matrix:")
print(df[telemetry_cols].corr().round(2))