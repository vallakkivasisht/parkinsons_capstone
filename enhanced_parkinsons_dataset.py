import os
import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# ---------- BASIC FEATURES ----------
def rms(series):
    return np.sqrt(np.mean(series**2))

def cv(series):
    return np.std(series) / np.mean(series) if np.mean(series) != 0 else 0

def avg_frequency(series, sampling_rate=50):  
    N = len(series)
    yf = np.abs(rfft(series - np.mean(series)))
    xf = rfftfreq(N, 1 / sampling_rate)
    return np.sum(xf * yf) / np.sum(yf) if np.sum(yf) != 0 else 0

# ---------- TOP 4 ADVANCED FEATURES FOR PARKINSON'S ----------
def tremor_power_ratio(series, sampling_rate=50):
    """
    FEATURE 1: Tremor Power Ratio (3-6 Hz / Total Power)
    Most discriminative for Parkinson's tremor
    """
    freqs, psd = signal.periodogram(series, fs=sampling_rate)
    
    # Total power (excluding DC)
    total_power = np.sum(psd[1:]) if len(psd) > 1 else 0
    
    # Tremor band power (3-6 Hz)
    tremor_mask = (freqs >= 3) & (freqs <= 6)
    tremor_power = np.sum(psd[tremor_mask])
    
    return tremor_power / total_power if total_power > 0 else 0

def dominant_frequency(series, sampling_rate=50):
    """
    FEATURE 2: Dominant Frequency
    Parkinson's patients show shifted dominant frequencies
    """
    freqs, psd = signal.periodogram(series, fs=sampling_rate)
    
    # Exclude DC component
    freqs = freqs[1:]
    psd = psd[1:]
    
    if len(psd) == 0:
        return 0
    
    return freqs[np.argmax(psd)]

def zero_crossing_rate(series):
    """
    FEATURE 3: Zero Crossing Rate
    Measures oscillatory nature - higher in tremor
    """
    if len(series) <= 1:
        return 0
    
    # Remove DC component
    series_centered = series - np.mean(series)
    
    # Count zero crossings
    zero_crossings = np.where(np.diff(np.sign(series_centered)))[0]
    
    return len(zero_crossings) / len(series)

def jerk_metric(series, sampling_rate=50):
    """
    FEATURE 4: Jerk (rate of acceleration change)
    Higher jerk indicates less smooth movement
    """
    if len(series) < 3:
        return 0
    
    # Calculate jerk (third derivative)
    dt = 1.0 / sampling_rate
    velocity = np.gradient(series, dt)
    acceleration = np.gradient(velocity, dt)
    jerk = np.gradient(acceleration, dt)
    
    # Return RMS of jerk
    return np.sqrt(np.mean(jerk**2))

# ---------- OPTIMIZED FEATURE EXTRACTION ----------
def extract_top10_features(file_path, sampling_rate=50):
    """
    Extract top 10 features most relevant for Parkinson's detection
    
    Features selected based on:
    1. Research evidence for Parkinson's detection
    2. Clinical relevance to tremor/bradykinesia/rigidity
    3. Computational efficiency
    4. Discriminative power
    """
    try:
        df = pd.read_csv(file_path)
        
        if df.shape[1] < 7:
            return None
        
        # Extract raw data
        acc = df.iloc[:, 1:4].values  # acc_x, acc_y, acc_z
        gyro = df.iloc[:, 4:7].values # gyro_x, gyro_y, gyro_z
        
        if len(acc) < 10:
            return None
        
        # Calculate magnitudes
        acc_mag = np.sqrt((acc**2).sum(axis=1))
        gyro_mag = np.sqrt((gyro**2).sum(axis=1))
        
        # TOP 10 FEATURES FOR PARKINSON'S DETECTION
        features = {
            # Original proven features (6 features)
            "ACC_RMS": rms(acc_mag),                    # Movement intensity
            "ACC_CV": cv(acc_mag),                      # Movement variability
            "ACC_AvgFreq": avg_frequency(acc_mag, sampling_rate),  # Spectral centroid
            "GYRO_RMS": rms(gyro_mag),                  # Rotational intensity
            "GYRO_CV": cv(gyro_mag),                    # Rotational variability
            "GYRO_AvgFreq": avg_frequency(gyro_mag, sampling_rate), # Rotational frequency
            
            # Advanced tremor-specific features (4 features)
            "ACC_TremorRatio": tremor_power_ratio(acc_mag, sampling_rate),    # Tremor dominance
            "GYRO_TremorRatio": tremor_power_ratio(gyro_mag, sampling_rate),  # Rotational tremor
            "ACC_DominantFreq": dominant_frequency(acc_mag, sampling_rate),   # Peak frequency
            "ACC_Jerk": jerk_metric(acc_mag, sampling_rate)                   # Movement smoothness
        }
        
        return features
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# ---------- MAIN PROCESSING ----------
def main():
    patients_csv = "patients_info_encoded.csv"
    base_folder = "timenew1"  # Change this to your folder name
    
    print("🎯 TOP 10 PARKINSON'S FEATURES EXTRACTION")
    print("="*50)
    print("Selected Features:")
    print("1-3. ACC_RMS, ACC_CV, ACC_AvgFreq (movement patterns)")
    print("4-6. GYRO_RMS, GYRO_CV, GYRO_AvgFreq (rotation patterns)")
    print("7-8. ACC/GYRO_TremorRatio (tremor detection)")
    print("9.   ACC_DominantFreq (frequency analysis)")
    print("10.  ACC_Jerk (movement smoothness)")
    print("="*50)
    
    try:
        patients_df = pd.read_csv(patients_csv)
    except FileNotFoundError:
        print(f"Error: Could not find {patients_csv}")
        return
    
    all_patient_features = []
    total_files = 0
    processed_files = 0
    
    # Process each patient
    for patient_idx, (_, patient) in enumerate(patients_df.iterrows(), 1):
        pid = int(patient["PatientID"])
        pid_str = str(pid).zfill(3)
        
        print(f"\nProcessing Patient {pid_str} ({patient_idx}/{len(patients_df)})")
        
        patient_folder = os.path.join(base_folder, pid_str)
        if not os.path.exists(patient_folder):
            print(f"  No folder found for patient {pid_str}")
            continue
        
        # Get CSV files
        patient_files = [f for f in os.listdir(patient_folder) if f.endswith(".csv")]
        
        if not patient_files:
            continue
        
        print(f"  Found {len(patient_files)} files")
        
        feature_list = []
        for file in sorted(patient_files):
            file_path = os.path.join(patient_folder, file)
            total_files += 1
            
            # Extract features
            features = extract_top10_features(file_path)
            
            if features is not None:
                features["File"] = file
                features["PatientID"] = pid
                feature_list.append(features)
                processed_files += 1
                print(f"    ✓ {file}")
            else:
                print(f"    ✗ {file}")
        
        # Add patient metadata
        if feature_list:
            patient_features_df = pd.DataFrame(feature_list)
            
            # Merge patient information
            for col in patients_df.columns:
                if col != "PatientID":
                    patient_features_df[col] = patient[col]
            
            all_patient_features.append(patient_features_df)
            print(f"  ✓ Patient {pid_str}: {len(feature_list)} files processed")
    
    # Save results
    if all_patient_features:
        final_df = pd.concat(all_patient_features, ignore_index=True)
        
        output_file = "patients_top10_features.csv"
        final_df.to_csv(output_file, index=False)
        
        print(f"\n" + "="*50)
        print("PROCESSING COMPLETE!")
        print(f"✓ Files processed: {processed_files}/{total_files} ({processed_files/total_files*100:.1f}%)")
        print(f"✓ Total records: {len(final_df)}")
        print(f"✓ Unique patients: {final_df['PatientID'].nunique()}")
        print(f"✓ Features extracted: 10")
        print(f"✓ Output saved: {output_file}")
        
        # Feature summary
        print(f"\nFeature Statistics:")
        feature_cols = ["ACC_RMS", "ACC_CV", "ACC_AvgFreq", "GYRO_RMS", "GYRO_CV", 
                       "GYRO_AvgFreq", "ACC_TremorRatio", "GYRO_TremorRatio", 
                       "ACC_DominantFreq", "ACC_Jerk"]
        
        for feat in feature_cols:
            if feat in final_df.columns:
                mean_val = final_df[feat].mean()
                std_val = final_df[feat].std()
                print(f"  {feat:<18}: {mean_val:.4f} ± {std_val:.4f}")
        
        print(f"\n🎯 Top 10 features ready for machine learning!")
        
    else:
        print("No features extracted. Check your folder paths and CSV files.")

if __name__ == "__main__":
    main()