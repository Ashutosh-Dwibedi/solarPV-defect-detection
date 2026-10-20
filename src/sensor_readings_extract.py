from pathlib import Path
import pandas as pd
import numpy as np

SRC_DIR = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
PROJECT_ROOT = SRC_DIR.parent if SRC_DIR.name in ["src"] else SRC_DIR

DATA_DIR = PROJECT_ROOT / "data"
LABELS_PATH = DATA_DIR / "labels.csv"
TELEMETRY_PATH = DATA_DIR / "paired_telemetry.csv"

df = pd.read_csv(LABELS_PATH)
df.columns = df.columns.str.strip()

d = df["defect_prob"].values
N = len(df)

np.random.seed(42)

# Genmerating Telemetry based on defect probability
voc = 0.61 - (0.08 * d) + np.random.normal(0, 0.0048, size=N)
isc = 8.85 - (1.25 * d) + np.random.normal(0, 0.0285, size=N)
rsh = np.exp(7.5 - 3.2 * d) + np.random.normal(0, 4.75, size=N)
# Since resistance cannot be negative, clipping the min at 0.1 Ohms.
rsh = np.clip(rsh, a_min=0.1, a_max=None)
temp = 25.0 + (14.5 * d) + np.random.normal(0, 0.76, size=N)

# Appending to DataFrame
df["voc"] = np.round(voc, 4)
df["isc"] = np.round(isc, 4)
df["rsh"] = np.round(rsh, 2)
df["temp"] = np.round(temp, 2)

# Saving the csv file
df.to_csv(TELEMETRY_PATH, index=False)