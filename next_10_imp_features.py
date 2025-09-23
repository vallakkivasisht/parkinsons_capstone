import os
import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy import signal
from scipy.stats import entropy, skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# ---------- NEXT 10 BEST FEATURES FOR PARKINSON'S ----------

def spectral_entropy(series, sampling_rate=50):
    """
    FEATURE 11: Spectral Entropy
    Measures complexity/irregularity of frequency spectrum
    Higher entropy = more irregular movement patterns
    """
    freqs, psd = signal.periodogram(series, fs=sampling_rate)
    
    # Normalize PSD to probability distribution
    psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
    psd_norm = psd_norm[psd_norm > 0]  # Remove zeros
    
    if len(psd_norm) <= 1:
        return 0
    
    return entropy(psd_norm, base=2)

def peak_rate(series, sampling_rate=50):
    """
    FEATURE 12: Peak Detection Rate
    Measures oscillatory activity - tremor creates regular peaks
    """
    if len(series) < 5:
        return 0
    
    # Use adaptive threshold
    threshold = np.mean(series) + 0.5 * np.std(series)
    peaks, _ = signal.find_peaks(series, height=threshold, distance=sampling_rate//10)
    
    return len(peaks) / (len(series) / sampling_rate)  # peaks per second

def frequency_dispersion(series, sampling_rate=50):
    """
    FEATURE 13: Frequency Dispersion (Spectral Spread)
    Measures how spread out the frequency content is
    Parkinson's may show more concentrated vs dispersed spectra
    """
    freqs, psd = signal.periodogram(series, fs=sampling_rate)
    freqs = freqs[1:]  # Remove DC
    psd = psd[1:]
    
    if np.sum(psd) == 0:
        return 0
    
    # Calculate spectral centroid (weighted mean frequency)
    spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
    
    # Calculate spectral spread (weighted standard deviation)
    spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd))
    
    return spectral_spread

def sample_entropy(series, m=2, r=None):
    """
    FEATURE 14: Sample Entropy
    Measures time series regularity/complexity
    Lower = more regular, Higher = more complex
    """
    series = np.array(series)
    N = len(series)
    
    if N < m + 1 or N < 10:
        return 0
    
    if r is None:
        r = 0.2 * np.std(series)
    
    def _maxdist(xi, xj):
        return max([abs(ua - va) for ua, va in zip(xi, xj)])
    
    def _phi(m):
        patterns = [series[i:i + m] for i in range(N - m + 1)]
        C = []
        
        for i in range(N - m + 1):
            template = patterns[i]
            matches = sum(1 for pattern in patterns if _maxdist(template, pattern) <= r)
            C.append(matches / (N - m + 1.0))
        
        phi = sum(np.log(c) for c in C if c > 0) / len(C) if len(C) > 0 else 0
        return phi
    
    try:
        return _phi(m) - _phi(m + 1)
    except:
        return 0

def movement_symmetry(acc_data):
    """
    FEATURE 15: Movement Symmetry
    Measures asymmetry between different movement directions
    Parkinson's often shows asymmetric movement patterns
    """
    if acc_data.shape[0] < 3 or acc_data.shape[1] != 3:
        return 0
    
    # Calculate RMS for each axis
    rms_x = np.sqrt(np.mean(acc_data[:, 0]**2))
    rms_y = np.sqrt(np.mean(acc_data[:, 1]**2))  
    rms_z = np.sqrt(np.mean(acc_data[:, 2]**2))
    
    # Calculate asymmetry as coefficient of variation of RMS values
    rms_values = [rms_x, rms_y, rms_z]
    mean_rms = np.mean(rms_values)
    
    if mean_rms == 0:
        return 0
    
    return np.std(rms_values) / mean_rms

def high_frequency_power(series, sampling_rate=50):
    """
    FEATURE 16: High Frequency Power Ratio (8-25 Hz)
    Measures high-frequency content that may indicate dyskinesia
    or other movement artifacts
    """
    freqs, psd = signal.periodogram(series, fs=sampling_rate)
    
    # Total power (excluding DC)
    total_power = np.sum(psd[1:]) if len(psd) > 1 else 0
    
    # High frequency power (8-25 Hz)
    hf_mask = (freqs >= 8) & (freqs <= 25)
    hf_power = np.sum(psd[hf_mask])
    
    return hf_power / total_power if total_power > 0 else 0

def acceleration_magnitude_variance(series):
    """
    FEATURE 17: Acceleration Magnitude Variance
    Measures variability in movement intensity
    Different from CV - focuses on magnitude fluctuations
    """
    if len(series) < 2:
        return 0
    
    return np.var(series)

def frequency_stability(series, sampling_rate=50, window_size=None):
    """
    FEATURE 18: Frequency Stability
    Measures how stable the dominant frequency is over time
    Parkinson's tremor is typically more stable than other movements
    """
    if window_size is None:
        window_size = min(len(series) // 4, sampling_rate * 2)  # 2-second windows or 1/4 of signal
    
    if len(series) < window_size * 2:
        return 0
    
    # Split signal into overlapping windows
    hop_size = window_size // 2
    dominant_freqs = []
    
    for i in range(0, len(series) - window_size, hop_size):
        window = series[i:i + window_size]
        freqs, psd = signal.periodogram(window, fs=sampling_rate)
        
        if len(psd) > 1:
            freqs = freqs[1:]  # Remove DC
            psd = psd[1:]
            if len(psd) > 0:
                dominant_freq = freqs[np.argmax(psd)]
                dominant_freqs.append(dominant_freq)
    
    if len(dominant_freqs) < 2:
        return 0
    
    # Calculate stability as inverse of coefficient of variation
    mean_freq = np.mean(dominant_freqs)
    if mean_freq == 0:
        return 0
    
    stability = 1 / (np.std(dominant_freqs) / mean_freq + 1e-6)
    return min(stability, 100)  # Cap at reasonable value

def cross_correlation_max(data1, data2):
    """
    FEATURE 19: Maximum Cross-Correlation between X and Y axes
    Measures coordination between different movement directions
    """
    if len(data1) != len(data2) or len(data1) < 5:
        return 0
    
    # Remove DC and normalize
    data1_norm = (data1 - np.mean(data1)) / (np.std(data1) + 1e-6)
    data2_norm = (data2 - np.mean(data2)) / (np.std(data2) + 1e-6)
    
    # Calculate cross-correlation
    correlation = np.correlate(data1_norm, data2_norm, mode='full')
    
    # Return maximum absolute correlation
    return np.max(np.abs(correlation)) / len(data1)

def movement_arrest_periods(series, threshold_ratio=0.1):
    """
    FEATURE 20: Movement Arrest Detection
    Measures periods of very low movement (related to freezing episodes)
    """
    if len(series) < 10:
        return 0
    
    # Define threshold as percentage of RMS
    rms_val = np.sqrt(np.mean(series**2))
    threshold = threshold_ratio * rms_val
    
    # Find periods below threshold
    below_threshold = series < threshold
    
    # Count consecutive periods below threshold
    arrest_periods = 0
    in_arrest = False
    current_arrest_length = 0
    
    for val in below_threshold:
        if val:  # Below threshold
            if not in_arrest:
                in_arrest = True
                current_arrest_length = 1
            else:
                current_arrest_length += 1
        else:  # Above threshold
            if in_arrest and current_arrest_length >= 3:  # Minimum 3 samples for arrest
                arrest_periods += 1
            in_arrest = False
            current_arrest_length = 0
    
    # Check if we ended in an arrest period
    if in_arrest and current_arrest_length >= 3:
        arrest_periods += 1
    
    return arrest_periods / len(series)  # Normalized by signal length

# ---------- FEATURE EXTRACTION ----------
def extract_next10_features(file_path, sampling_rate=50):
    """
    Extract the next 10 most valuable Parkinson's detection features
    
    These complement the first 10 features by focusing on:
    - Spectral complexity and stability
    - Movement coordination and symmetry  
    - Temporal patterns and arrest detection
    - High-frequency content analysis
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
        
        # NEXT 10 FEATURES FOR PARKINSON'S DETECTION
        features = {
            # Spectral complexity features
            "ACC_SpectralEntropy": spectral_entropy(acc_mag, sampling_rate),      # Feature 11
            "GYRO_SpectralEntropy": spectral_entropy(gyro_mag, sampling_rate),    # Feature 12
            
            # Oscillatory activity
            "ACC_PeakRate": peak_rate(acc_mag, sampling_rate),                     # Feature 13
            "GYRO_PeakRate": peak_rate(gyro_mag, sampling_rate),                   # Feature 14
            
            # Frequency characteristics
            "ACC_FreqDispersion": frequency_dispersion(acc_mag, sampling_rate),   # Feature 15
            "ACC_FreqStability": frequency_stability(acc_mag, sampling_rate),     # Feature 16
            
            # Movement patterns
            "ACC_SampleEntropy": sample_entropy(acc_mag),                         # Feature 17
            "ACC_MovementSymmetry": movement_symmetry(acc),                       # Feature 18
            
            # Advanced spectral features
            "ACC_HighFreqRatio": high_frequency_power(acc_mag, sampling_rate),    # Feature 19
            "ACC_MovementArrests": movement_arrest_periods(acc_mag)               # Feature 20
        }
        
        return features
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# ---------- MAIN PROCESSING ----------
def main():
    patients_csv = "patients_info_encoded.csv"
    base_folder = "timenew1"  # Change this to your folder name
    
    print("NEXT 10 PARKINSON'S FEATURES EXTRACTION")
    print("="*50)
    print("Features 11-20:")
    print("11-12. Spectral Entropy (complexity)")
    print("13-14. Peak Rate (oscillatory activity)")
    print("15.    Frequency Dispersion (spectral spread)")
    print("16.    Frequency Stability (tremor consistency)")
    print("17.    Sample Entropy (time series complexity)")
    print("18.    Movement Symmetry (axis coordination)")
    print("19.    High Frequency Ratio (dyskinesia detection)")
    print("20.    Movement Arrests (freezing episodes)")
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
            features = extract_next10_features(file_path)
            
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
        
        output_file = "patients_next10_features.csv"
        final_df.to_csv(output_file, index=False)
        
        print(f"\n" + "="*50)
        print("PROCESSING COMPLETE!")
        print(f"✓ Files processed: {processed_files}/{total_files} ({processed_files/total_files*100:.1f}%)")
        print(f"✓ Total records: {len(final_df)}")
        print(f"✓ Unique patients: {final_df['PatientID'].nunique()}")
        print(f"✓ Features extracted: 10 additional")
        print(f"✓ Output saved: {output_file}")
        
        # Feature summary
        print(f"\nFeature Statistics:")
        feature_cols = ["ACC_SpectralEntropy", "GYRO_SpectralEntropy", "ACC_PeakRate", 
                       "GYRO_PeakRate", "ACC_FreqDispersion", "ACC_FreqStability",
                       "ACC_SampleEntropy", "ACC_MovementSymmetry", "ACC_HighFreqRatio", 
                       "ACC_MovementArrests"]
        
        for feat in feature_cols:
            if feat in final_df.columns:
                mean_val = final_df[feat].mean()
                std_val = final_df[feat].std()
                print(f"  {feat:<20}: {mean_val:.4f} ± {std_val:.4f}")
        
        print(f"\nNext 10 features ready! Combine with your first 10 for 20 total features.")
        print("To merge datasets:")
        print("df1 = pd.read_csv('patients_top10_features.csv')")
        print("df2 = pd.read_csv('patients_next10_features.csv')")
        print("df_combined = df1.merge(df2, on=['PatientID', 'File'])")
        
    else:
        print("No features extracted. Check your folder paths and CSV files.")

if __name__ == "__main__":
    main()