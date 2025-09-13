import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq

# Load your file
file_path = r"C:\Users\vallakki Vasisht\OneDrive\Documents\capstone\working_model\timenew1\011\011_HoldWeight_RightWrist.csv"
df = pd.read_csv(file_path)

# Extract accelerometer and gyroscope values
acc = df.iloc[:, 1:4].values  # acc_x, acc_y, acc_z
gyro = df.iloc[:, 4:7].values # gyro_x, gyro_y, gyro_z

# Compute magnitudes
acc_mag = np.sqrt((acc**2).sum(axis=1))
gyro_mag = np.sqrt((gyro**2).sum(axis=1))

# Functions
def rms(series):
    return np.sqrt(np.mean(series**2))

def cv(series):
    mean_val = np.mean(series)
    return np.std(series) / mean_val if mean_val != 0 else 0

def avg_frequency(series, sampling_rate=50):
    N = len(series)
    series = series - np.mean(series)  # remove DC
    yf = np.abs(rfft(series))**2       # power spectrum
    xf = rfftfreq(N, 1 / sampling_rate)
    xf = xf[1:]
    yf = yf[1:]
    return np.sum(xf * yf) / np.sum(yf) if np.sum(yf) != 0 else 0

# Calculate features
results = {
    "ACC_RMS": rms(acc_mag),
    "ACC_CV": cv(acc_mag),
    "ACC_AvgFreq": avg_frequency(acc_mag),
    "GYRO_RMS": rms(gyro_mag),
    "GYRO_CV": cv(gyro_mag),
    "GYRO_AvgFreq": avg_frequency(gyro_mag)
}

print(results)
