import pandas as pd
import numpy as np
from scipy.stats import entropy

# Load dataset
df = pd.read_csv("patients_top10_features.csv")

# ---- Feature 1: Jerk (rate of change of ACC_RMS) ----
if "ACC_RMS" in df.columns:
    df["ACC_Jerk"] = df["ACC_RMS"].diff().fillna(0).apply(lambda x: abs(x))

# ---- Feature 2: Entropy (signal complexity on ACC_RMS & GYRO_RMS) ----
def compute_entropy(series):
    # normalize histogram for entropy
    hist, _ = np.histogram(series, bins=20, density=True)
    hist = hist[hist > 0]  # remove zero bins
    return entropy(hist)

if "ACC_RMS" in df.columns:
    df["ACC_Entropy"] = compute_entropy(df["ACC_RMS"].values)

if "GYRO_RMS" in df.columns:
    df["GYRO_Entropy"] = compute_entropy(df["GYRO_RMS"].values)

# Save updated dataset
df.to_csv("patients_features_with_new.csv", index=False)
print("✅ Added ACC_Jerk and Entropy features. Saved as patients_features_with_new.csv")
