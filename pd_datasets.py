import os
import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq

# ---------- FUNCTIONS ----------
def rms(series):
    return np.sqrt(np.mean(series**2))

def cv(series):
    return np.std(series) / np.mean(series) if np.mean(series) != 0 else 0

def avg_frequency(series, sampling_rate=50):  
    """Compute average frequency using power-weighted FFT magnitude."""
    N = len(series)
    series = series - np.mean(series)  # remove DC
    yf = np.abs(rfft(series))**2       # power spectrum
    xf = rfftfreq(N, 1 / sampling_rate)

    # Avoid DC component at 0 Hz
    xf = xf[1:]
    yf = yf[1:]

    return np.sum(xf * yf) / np.sum(yf) if np.sum(yf) != 0 else 0

def extract_features(file_path, sampling_rate=50):
    df = pd.read_csv(file_path)
    acc = df.iloc[:, 1:4].values  # acc_x, acc_y, acc_z
    gyro = df.iloc[:, 4:7].values # gyro_x, gyro_y, gyro_z

    acc_mag = np.sqrt((acc**2).sum(axis=1))
    gyro_mag = np.sqrt((gyro**2).sum(axis=1))

    return {
        "ACC_RMS": rms(acc_mag),
        "ACC_CV": cv(acc_mag),
        "ACC_AvgFreq": avg_frequency(acc_mag, sampling_rate),
        "GYRO_RMS": rms(gyro_mag),
        "GYRO_CV": cv(gyro_mag),
        "GYRO_AvgFreq": avg_frequency(gyro_mag, sampling_rate),
        "FileName": os.path.basename(file_path)   # ✅ new column with CSV file name
    }

# ---------- MAIN ----------
patients_csv = "patients_info_encoded.csv"   # your main patient info CSV
base_folder = "timenew1"                     # main folder containing patient subfolders

patients_df = pd.read_csv(patients_csv)
all_patient_features = []

# loop through all patients
for _, patient in patients_df.iterrows():
    pid = int(patient["PatientID"])             # numeric patient ID (1,2,3,...)
    pid_str = str(pid).zfill(3)                 # convert to "001", "002", ...

    patient_folder = os.path.join(base_folder, pid_str)
    if not os.path.exists(patient_folder):
        print(f"⚠️ No folder found for patient {pid_str}")
        continue

    # get all csv files inside the patient's folder
    patient_files = [f for f in os.listdir(patient_folder) if f.endswith(".csv")]

    feature_list = []
    for file in sorted(patient_files):
        file_path = os.path.join(patient_folder, file)
        try:
            feats = extract_features(file_path)
            feats["File"] = file
            feats["PatientID"] = pid        # keep numeric ID in final CSV
            feature_list.append(feats)
            print(f"✅ Processed {file}")
        except Exception as e:
            print(f"⚠️ Skipped {file}: {e}")

    if feature_list:
        patient_features_df = pd.DataFrame(feature_list)

        # merge patient metadata to all rows
        for col in patients_df.columns:
            if col != "PatientID":
                patient_features_df[col] = patient[col]

        all_patient_features.append(patient_features_df)

# combine everything
if all_patient_features:
    final_df = pd.concat(all_patient_features, ignore_index=True)
    final_df.to_csv("patients_features_merged.csv", index=False)
    print("🎯 Done! Saved as patients_features_merged.csv")
else:
    print("⚠️ No features extracted. Please check your folder paths and CSV files.")
