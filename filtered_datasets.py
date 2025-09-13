import os
import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy import signal

# ---------- FILTERING FUNCTIONS ----------
def bandpass_filter(data, low_freq=0.5, high_freq=20, sampling_rate=50, order=4):
    """
    Apply bandpass filter to remove noise and isolate movement frequencies
    
    Parameters:
    - data: input signal
    - low_freq: lower cutoff frequency (Hz) - removes very slow drifts
    - high_freq: upper cutoff frequency (Hz) - removes high-frequency noise
    - sampling_rate: sampling frequency (Hz)
    - order: filter order (higher = sharper cutoff)
    
    Returns:
    - filtered_data: bandpass filtered signal
    """
    # Calculate Nyquist frequency
    nyquist = sampling_rate / 2
    
    # Normalize frequencies
    low_norm = low_freq / nyquist
    high_norm = high_freq / nyquist
    
    # Ensure frequencies are within valid range
    low_norm = max(0.001, min(low_norm, 0.999))  # Avoid exactly 0 or 1
    high_norm = max(low_norm + 0.001, min(high_norm, 0.999))
    
    try:
        # Design Butterworth bandpass filter
        b, a = signal.butter(order, [low_norm, high_norm], btype='band', analog=False)
        
        # Apply filter (use filtfilt for zero-phase filtering)
        filtered_data = signal.filtfilt(b, a, data)
        
        return filtered_data
    
    except Exception as e:
        print(f"⚠️ Filtering failed: {e}")
        print(f"   Low freq: {low_freq} Hz, High freq: {high_freq} Hz")
        print(f"   Normalized: {low_norm:.3f} - {high_norm:.3f}")
        return data  # Return original data if filtering fails

def apply_bandpass_to_imu(acc_data, gyro_data, sampling_rate=50):
    """
    Apply appropriate bandpass filters to accelerometer and gyroscope data
    
    Different frequency ranges for different sensors:
    - Accelerometer: 0.5-15 Hz (human movement + tremor range)
    - Gyroscope: 0.3-20 Hz (rotational movements, slightly wider range)
    """
    
    # Filter accelerometer data (each axis)
    acc_filtered = np.zeros_like(acc_data)
    for i in range(acc_data.shape[1]):  # For each axis (x, y, z)
        acc_filtered[:, i] = bandpass_filter(
            acc_data[:, i], 
            low_freq=0.5,   # Remove very slow drifts/gravity components
            high_freq=15,   # Remove high-frequency noise, keep tremor range
            sampling_rate=sampling_rate
        )
    
    # Filter gyroscope data (each axis)  
    gyro_filtered = np.zeros_like(gyro_data)
    for i in range(gyro_data.shape[1]):  # For each axis (x, y, z)
        gyro_filtered[:, i] = bandpass_filter(
            gyro_data[:, i],
            low_freq=0.3,   # Lower cutoff for rotational movements
            high_freq=20,   # Higher cutoff to capture faster rotations
            sampling_rate=sampling_rate
        )
    
    return acc_filtered, gyro_filtered

# ---------- FEATURE EXTRACTION FUNCTIONS ----------
def rms(series):
    return np.sqrt(np.mean(series**2))

def cv(series):
    return np.std(series) / np.mean(series) if np.mean(series) != 0 else 0

def avg_frequency(series, sampling_rate=50):  
    """Compute average frequency using power-weighted FFT magnitude (Spectral Centroid)."""
    N = len(series)
    series = series - np.mean(series)  # remove DC
    yf = np.abs(rfft(series))**2       # power spectrum
    xf = rfftfreq(N, 1 / sampling_rate)

    # Avoid DC component at 0 Hz
    xf = xf[1:]
    yf = yf[1:]

    return np.sum(xf * yf) / np.sum(yf) if np.sum(yf) != 0 else 0

def extract_features_filtered(file_path, sampling_rate=50):
    """
    Extract features from FILTERED IMU data
    """
    try:
        df = pd.read_csv(file_path)
        
        # Check if we have enough columns
        if df.shape[1] < 7:
            print(f"⚠️ Not enough columns in {file_path}: {df.shape[1]}")
            return None
            
        # Extract raw IMU data
        acc_raw = df.iloc[:, 1:4].values  # acc_x, acc_y, acc_z
        gyro_raw = df.iloc[:, 4:7].values # gyro_x, gyro_y, gyro_z
        
        # Check for minimum data length (need enough samples for filtering)
        min_samples = 10
        if len(acc_raw) < min_samples:
            print(f"⚠️ Not enough samples in {file_path}: {len(acc_raw)}")
            return None
        
        # Apply bandpass filtering
        acc_filtered, gyro_filtered = apply_bandpass_to_imu(acc_raw, gyro_raw, sampling_rate)
        
        # Calculate magnitudes from FILTERED data
        acc_mag_filtered = np.sqrt((acc_filtered**2).sum(axis=1))
        gyro_mag_filtered = np.sqrt((gyro_filtered**2).sum(axis=1))
        
        # Extract features from filtered magnitudes
        features = {
            # Features from FILTERED data
            "ACC_RMS_filtered": rms(acc_mag_filtered),
            "ACC_CV_filtered": cv(acc_mag_filtered),
            "ACC_AvgFreq_filtered": avg_frequency(acc_mag_filtered, sampling_rate),
            "GYRO_RMS_filtered": rms(gyro_mag_filtered),
            "GYRO_CV_filtered": cv(gyro_mag_filtered),
            "GYRO_AvgFreq_filtered": avg_frequency(gyro_mag_filtered, sampling_rate),
            
            # Optional: Features from RAW data for comparison
            "ACC_RMS_raw": rms(np.sqrt((acc_raw**2).sum(axis=1))),
            "ACC_CV_raw": cv(np.sqrt((acc_raw**2).sum(axis=1))),
            "ACC_AvgFreq_raw": avg_frequency(np.sqrt((acc_raw**2).sum(axis=1)), sampling_rate),
            "GYRO_RMS_raw": rms(np.sqrt((gyro_raw**2).sum(axis=1))),
            "GYRO_CV_raw": cv(np.sqrt((gyro_raw**2).sum(axis=1))),
            "GYRO_AvgFreq_raw": avg_frequency(np.sqrt((gyro_raw**2).sum(axis=1)), sampling_rate),
            
            # Metadata
            "FileName": os.path.basename(file_path),
            "SampleCount": len(acc_raw),
            "FilterApplied": True
        }
        
        return features
        
    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")
        return None

# ---------- MAIN PROCESSING ----------
patients_csv = "patients_info_encoded.csv"   
base_folder = "timenew1"                     

print("🚀 Starting FILTERED feature extraction...")
print("="*50)

# Load patient information
patients_df = pd.read_csv(patients_csv)
print(f"✅ Loaded {len(patients_df)} patients")

all_patient_features = []
total_files = 0
processed_files = 0
failed_files = 0

# Loop through all patients
for patient_idx, (_, patient) in enumerate(patients_df.iterrows(), 1):
    pid = int(patient["PatientID"])
    pid_str = str(pid).zfill(3)
    
    print(f"\n📁 Processing Patient {pid_str} ({patient_idx}/{len(patients_df)})")
    
    patient_folder = os.path.join(base_folder, pid_str)
    if not os.path.exists(patient_folder):
        print(f"⚠️ No folder found for patient {pid_str}")
        continue

    # Get all CSV files for this patient
    patient_files = [f for f in os.listdir(patient_folder) if f.endswith(".csv")]
    
    if not patient_files:
        print(f"⚠️ No CSV files found for patient {pid_str}")
        continue
    
    print(f"   Found {len(patient_files)} files")
    
    feature_list = []
    for file in sorted(patient_files):
        file_path = os.path.join(patient_folder, file)
        total_files += 1
        
        # Extract features with filtering
        features = extract_features_filtered(file_path)
        
        if features is not None:
            features["File"] = file
            features["PatientID"] = pid
            feature_list.append(features)
            processed_files += 1
            print(f"   ✅ {file}")
        else:
            failed_files += 1
            print(f"   ❌ {file}")

    # Add patient metadata to all feature records
    if feature_list:
        patient_features_df = pd.DataFrame(feature_list)
        
        # Merge patient info (age, condition, etc.)
        for col in patients_df.columns:
            if col != "PatientID":
                patient_features_df[col] = patient[col]

        all_patient_features.append(patient_features_df)
        print(f"   ✅ Patient {pid_str}: {len(feature_list)} files processed")
    else:
        print(f"   ❌ Patient {pid_str}: No files processed successfully")

# Combine all patient data
print("\n" + "="*50)
print("📊 PROCESSING SUMMARY")
print("="*50)

if all_patient_features:
    final_df = pd.concat(all_patient_features, ignore_index=True)
    
    # Save results
    output_file = "patients_features_filtered.csv"
    final_df.to_csv(output_file, index=False)
    
    print(f"✅ SUCCESS!")
    print(f"   Total files found: {total_files}")
    print(f"   Files processed: {processed_files}")
    print(f"   Files failed: {failed_files}")
    print(f"   Success rate: {processed_files/total_files*100:.1f}%")
    print(f"   Total records: {len(final_df)}")
    print(f"   Unique patients: {final_df['PatientID'].nunique()}")
    print(f"   Output saved: {output_file}")
    
    # Show sample of filtered vs raw features
    print(f"\n📈 SAMPLE COMPARISON (Filtered vs Raw):")
    print("-" * 40)
    sample_row = final_df.iloc[0]
    comparison_features = [
        ("ACC_RMS", "ACC_RMS_filtered", "ACC_RMS_raw"),
        ("ACC_AvgFreq", "ACC_AvgFreq_filtered", "ACC_AvgFreq_raw"),
        ("GYRO_RMS", "GYRO_RMS_filtered", "GYRO_RMS_raw")
    ]
    
    for feat_name, filtered_col, raw_col in comparison_features:
        filtered_val = sample_row[filtered_col]
        raw_val = sample_row[raw_col]
        change = ((filtered_val - raw_val) / raw_val * 100) if raw_val != 0 else 0
        print(f"{feat_name:<12}: Raw={raw_val:.4f}, Filtered={filtered_val:.4f} ({change:+.1f}%)")
    
    print(f"\n🎯 Ready for analysis!")
    print(f"   Main features: ACC/GYRO_RMS/CV/AvgFreq_filtered")
    print(f"   Comparison features: ACC/GYRO_RMS/CV/AvgFreq_raw")
    
else:
    print("❌ No features extracted. Please check:")
    print("   - Folder paths and file structure")
    print("   - CSV file format (7 columns expected)")
    print("   - Data quality and length")

print("="*50)
print("🏁 Processing complete!")