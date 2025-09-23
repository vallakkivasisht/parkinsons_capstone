import os
import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy import signal
from scipy.stats import skew, pearsonr
import warnings
warnings.filterwarnings('ignore')

# ---------- FOCUSED 15 HIGH-IMPACT FEATURES ----------

def tremor_power_ratio(data_axis, sampling_rate=50):
    """Calculate tremor band power ratio for individual axis"""
    freqs, psd = signal.periodogram(data_axis - np.mean(data_axis), fs=sampling_rate)
    total_power = np.sum(psd[1:]) if len(psd) > 1 else 0
    tremor_mask = (freqs >= 3) & (freqs <= 6)
    tremor_power = np.sum(psd[tremor_mask])
    return tremor_power / (total_power + 1e-10)

def dominant_frequency(data_axis, sampling_rate=50):
    """Get dominant frequency for individual axis"""
    freqs, psd = signal.periodogram(data_axis - np.mean(data_axis), fs=sampling_rate)
    if len(freqs) <= 1:
        return 0
    freqs = freqs[1:]  # Remove DC
    psd = psd[1:]
    return freqs[np.argmax(psd)] if len(psd) > 0 else 0

def extract_focused_15_features(file_path, sampling_rate=50):
    """
    Extract exactly 15 high-discriminative features for Parkinson's detection
    Focused on the most research-proven discriminative aspects
    """
    try:
        df = pd.read_csv(file_path)
        
        if df.shape[1] < 7:
            return None
        
        acc = df.iloc[:, 1:4].values  # acc_x, acc_y, acc_z
        gyro = df.iloc[:, 4:7].values # gyro_x, gyro_y, gyro_z
        
        if len(acc) < 10:
            return None
        
        # Calculate magnitudes for comparison
        acc_mag = np.sqrt((acc**2).sum(axis=1))
        gyro_mag = np.sqrt((gyro**2).sum(axis=1))
        
        features = {}
        
        # ========================================
        # FEATURES 1-6: INDIVIDUAL AXIS VARIABILITY
        # Research shows these have highest discriminative power
        # ========================================
        features['ACC_X_Std'] = np.std(acc[:, 0])      # X-axis movement variability
        features['ACC_Y_Std'] = np.std(acc[:, 1])      # Y-axis movement variability  
        features['ACC_Z_Std'] = np.std(acc[:, 2])      # Z-axis movement variability
        features['GYRO_X_Std'] = np.std(gyro[:, 0])    # X-axis rotation variability
        features['GYRO_Y_Std'] = np.std(gyro[:, 1])    # Y-axis rotation variability
        features['GYRO_Z_Std'] = np.std(gyro[:, 2])    # Z-axis rotation variability
        
        # ========================================
        # FEATURES 7-9: ASYMMETRY MEASURES
        # Parkinson's shows characteristic asymmetric patterns
        # ========================================
        features['ACC_X_Skewness'] = skew(acc[:, 0])   # Movement asymmetry in X
        features['ACC_Y_Skewness'] = skew(acc[:, 1])   # Movement asymmetry in Y
        features['GYRO_X_Skewness'] = skew(gyro[:, 0]) # Rotation asymmetry in X
        
        # ========================================
        # FEATURES 10-12: TREMOR-SPECIFIC (INDIVIDUAL AXES)
        # Individual axes preserve directional tremor information
        # ========================================
        features['ACC_X_TremorRatio'] = tremor_power_ratio(acc[:, 0], sampling_rate)
        features['ACC_Y_TremorRatio'] = tremor_power_ratio(acc[:, 1], sampling_rate) 
        features['GYRO_Z_TremorRatio'] = tremor_power_ratio(gyro[:, 2], sampling_rate)  # Z-axis rotation tremor
        
        # ========================================
        # FEATURES 13-14: COORDINATION MEASURES
        # Cross-axis correlations indicate movement coordination
        # ========================================
        try:
            features['ACC_XY_Correlation'] = pearsonr(acc[:, 0], acc[:, 1])[0]
            features['GYRO_XY_Correlation'] = pearsonr(gyro[:, 0], gyro[:, 1])[0]
        except:
            features['ACC_XY_Correlation'] = 0
            features['GYRO_XY_Correlation'] = 0
        
        # Replace NaN correlations with 0
        if np.isnan(features['ACC_XY_Correlation']):
            features['ACC_XY_Correlation'] = 0
        if np.isnan(features['GYRO_XY_Correlation']):
            features['GYRO_XY_Correlation'] = 0
        
        # ========================================
        # FEATURE 15: MAGNITUDE-BASED COMPARISON
        # Keep one magnitude feature for comparison
        # ========================================
        features['ACC_Mag_CV'] = np.std(acc_mag) / (np.mean(acc_mag) + 1e-10)
        
        return features
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# ---------- MAIN PROCESSING ----------
def main():
    patients_csv = "patients_info_encoded.csv"
    base_folder = "timenew1"  # Change this
    
    print("FOCUSED 15 HIGH-IMPACT FEATURES")
    print("="*40)
    print("Extracting exactly 15 features:")
    print("1-6.   Individual axis variability (Std)")
    print("7-9.   Movement asymmetry (Skewness)")
    print("10-12. Axis-specific tremor ratios")
    print("13-14. Cross-axis coordination")
    print("15.    Magnitude CV for comparison")
    print("="*40)
    
    try:
        patients_df = pd.read_csv(patients_csv)
    except FileNotFoundError:
        print(f"Error: Could not find {patients_csv}")
        return
    
    all_patient_features = []
    total_files = 0
    processed_files = 0
    
    for patient_idx, (_, patient) in enumerate(patients_df.iterrows(), 1):
        pid = int(patient["PatientID"])
        pid_str = str(pid).zfill(3)
        
        print(f"\nPatient {pid_str} ({patient_idx}/{len(patients_df)})")
        
        patient_folder = os.path.join(base_folder, pid_str)
        if not os.path.exists(patient_folder):
            continue
        
        patient_files = [f for f in os.listdir(patient_folder) if f.endswith(".csv")]
        if not patient_files:
            continue
        
        feature_list = []
        for file in sorted(patient_files):
            file_path = os.path.join(patient_folder, file)
            total_files += 1
            
            features = extract_focused_15_features(file_path)
            
            if features is not None:
                features["File"] = file
                features["PatientID"] = pid
                feature_list.append(features)
                processed_files += 1
        
        if feature_list:
            patient_features_df = pd.DataFrame(feature_list)
            
            for col in patients_df.columns:
                if col != "PatientID":
                    patient_features_df[col] = patient[col]
            
            all_patient_features.append(patient_features_df)
            print(f"  Processed: {len(feature_list)} files")
    
    # Save results
    if all_patient_features:
        final_df = pd.concat(all_patient_features, ignore_index=True)
        
        output_file = "patients_focused15_features.csv"
        final_df.to_csv(output_file, index=False)
        
        print(f"\n" + "="*40)
        print("EXTRACTION COMPLETE!")
        print(f"Files processed: {processed_files}/{total_files}")
        print(f"Total records: {len(final_df)}")
        print(f"Features: Exactly 15")
        print(f"Output: {output_file}")
        
        # Quick correlation check
        print(f"\nQuick correlation check with target:")
        try:
            from sklearn.preprocessing import LabelEncoder
            
            feature_cols = [col for col in final_df.columns 
                           if any(x in col for x in ['ACC_', 'GYRO_']) 
                           and col not in ['File', 'PatientID']]
            
            le = LabelEncoder()
            y_encoded = le.fit_transform(final_df['condition'])
            
            correlations = []
            for feature in feature_cols[:10]:  # Show first 10
                if feature in final_df.columns:
                    corr = np.corrcoef(final_df[feature].fillna(0), y_encoded)[0, 1]
                    correlations.append((feature, abs(corr)))
                    print(f"{feature:<20}: {corr:.4f}")
            
            # Find strongest correlation
            correlations.sort(key=lambda x: x[1], reverse=True)
            strongest = correlations[0] if correlations else ("None", 0)
            print(f"\nStrongest correlation: {strongest[0]} ({strongest[1]:.4f})")
            
        except Exception as e:
            print(f"Correlation check failed: {e}")
        
        print(f"\nREADY FOR MACHINE LEARNING!")
        print("These 15 features target the core discriminative aspects")
        print("without overwhelming your system.")
        
    else:
        print("No features extracted.")

if __name__ == "__main__":
    main()