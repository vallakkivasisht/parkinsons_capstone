import os
import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq

# ---------- EXACT SAME FUNCTIONS AS YOUR CODE ----------
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

def recalculate_features(file_path, sampling_rate=50):
    """Recalculate features from raw CSV file"""
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
        "GYRO_AvgFreq": avg_frequency(gyro_mag, sampling_rate)
    }

def validate_features():
    """Validate features by comparing stored values with recalculated ones"""
    
    # Load your generated feature dataset
    features_csv = "patients_features_merged.csv"
    base_folder = "timenew1"
    
    if not os.path.exists(features_csv):
        print(f"❌ Feature file '{features_csv}' not found!")
        return
    
    # Read the features dataset
    features_df = pd.read_csv(features_csv)
    print(f"📊 Loaded {len(features_df)} feature records")
    
    # Validation results
    validation_results = []
    total_records = len(features_df)
    
    print("\n🔍 Starting validation...")
    print("="*80)
    
    for idx, row in features_df.iterrows():
        patient_id = str(int(row['PatientID'])).zfill(3)
        file_name = row['File']
        
        # Construct file path
        file_path = os.path.join(base_folder, patient_id, file_name)
        
        if not os.path.exists(file_path):
            print(f"⚠️  File not found: {file_path}")
            continue
        
        try:
            # Recalculate features
            recalc_features = recalculate_features(file_path)
            
            # Compare values
            comparison = {
                'PatientID': row['PatientID'],
                'File': file_name,
                'Status': 'PASS'
            }
            
            tolerance = 1e-10  # Very small tolerance for floating point comparison
            
            for feature in ['ACC_RMS', 'ACC_CV', 'ACC_AvgFreq', 'GYRO_RMS', 'GYRO_CV', 'GYRO_AvgFreq']:
                stored_value = row[feature]
                recalc_value = recalc_features[feature]
                diff = abs(stored_value - recalc_value)
                
                comparison[f'{feature}_stored'] = stored_value
                comparison[f'{feature}_recalc'] = recalc_value
                comparison[f'{feature}_diff'] = diff
                comparison[f'{feature}_match'] = diff < tolerance
                
                if diff >= tolerance:
                    comparison['Status'] = 'FAIL'
            
            validation_results.append(comparison)
            
            # Print progress and results
            status_emoji = "✅" if comparison['Status'] == 'PASS' else "❌"
            print(f"{status_emoji} Patient {patient_id}, File: {file_name} - {comparison['Status']}")
            
            # If failed, show detailed comparison
            if comparison['Status'] == 'FAIL':
                print("   Detailed comparison:")
                for feature in ['ACC_RMS', 'ACC_CV', 'ACC_AvgFreq', 'GYRO_RMS', 'GYRO_CV', 'GYRO_AvgFreq']:
                    stored = comparison[f'{feature}_stored']
                    recalc = comparison[f'{feature}_recalc']
                    diff = comparison[f'{feature}_diff']
                    match_status = "✅" if comparison[f'{feature}_match'] else "❌"
                    print(f"     {match_status} {feature}: Stored={stored:.10f}, Recalc={recalc:.10f}, Diff={diff:.2e}")
                print()
            
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            validation_results.append({
                'PatientID': row['PatientID'],
                'File': file_name,
                'Status': 'ERROR',
                'Error': str(e)
            })
    
    # Summary statistics
    validation_df = pd.DataFrame(validation_results)
    
    if len(validation_df) > 0:
        total_validated = len(validation_df)
        passed = len(validation_df[validation_df['Status'] == 'PASS'])
        failed = len(validation_df[validation_df['Status'] == 'FAIL'])
        errors = len(validation_df[validation_df['Status'] == 'ERROR'])
        
        print("="*80)
        print("📈 VALIDATION SUMMARY")
        print("="*80)
        print(f"Total records in dataset: {total_records}")
        print(f"Records validated: {total_validated}")
        print(f"✅ Passed: {passed} ({passed/total_validated*100:.1f}%)")
        print(f"❌ Failed: {failed} ({failed/total_validated*100:.1f}%)")
        print(f"⚠️  Errors: {errors} ({errors/total_validated*100:.1f}%)")
        
        # Save detailed validation results
        validation_output = "feature_validation_results.csv"
        validation_df.to_csv(validation_output, index=False)
        print(f"\n💾 Detailed results saved to: {validation_output}")
        
        # Show feature-specific accuracy
        if failed > 0:
            print("\n🔍 FEATURE-SPECIFIC ANALYSIS:")
            print("-"*40)
            for feature in ['ACC_RMS', 'ACC_CV', 'ACC_AvgFreq', 'GYRO_RMS', 'GYRO_CV', 'GYRO_AvgFreq']:
                if f'{feature}_match' in validation_df.columns:
                    matches = validation_df[f'{feature}_match'].sum()
                    total = len(validation_df[validation_df['Status'].isin(['PASS', 'FAIL'])])
                    accuracy = matches / total * 100 if total > 0 else 0
                    status_emoji = "✅" if accuracy == 100 else "⚠️" if accuracy > 95 else "❌"
                    print(f"{status_emoji} {feature}: {matches}/{total} ({accuracy:.1f}%)")
        
        return validation_df
    else:
        print("❌ No records could be validated!")
        return None

# ---------- QUICK CHECK FUNCTION ----------
def quick_check_sample(patient_id="001", file_name=None, sampling_rate=50):
    """Quick check for a specific file"""
    base_folder = "timenew1"
    features_csv = "patients_features_merged.csv"
    
    # Load features dataset
    features_df = pd.read_csv(features_csv)
    
    # Filter for specific patient
    patient_data = features_df[features_df['PatientID'] == int(patient_id)]
    
    if len(patient_data) == 0:
        print(f"❌ No data found for patient {patient_id}")
        return
    
    # Use first file if none specified
    if file_name is None:
        file_name = patient_data.iloc[0]['File']
        print(f"📁 Using file: {file_name}")
    
    # Get stored values
    stored_row = patient_data[patient_data['File'] == file_name]
    if len(stored_row) == 0:
        print(f"❌ File {file_name} not found for patient {patient_id}")
        return
    
    stored_row = stored_row.iloc[0]
    
    # Recalculate features
    file_path = os.path.join(base_folder, patient_id, file_name)
    if not os.path.exists(file_path):
        print(f"❌ Raw file not found: {file_path}")
        return
    
    recalc_features = recalculate_features(file_path, sampling_rate)
    
    # Compare
    print(f"\n🔍 COMPARISON FOR PATIENT {patient_id}, FILE: {file_name}")
    print("="*60)
    print(f"{'Feature':<15} {'Stored':<15} {'Recalculated':<15} {'Match':<10}")
    print("-"*60)
    
    for feature in ['ACC_RMS', 'ACC_CV', 'ACC_AvgFreq', 'GYRO_RMS', 'GYRO_CV', 'GYRO_AvgFreq']:
        stored = stored_row[feature]
        recalc = recalc_features[feature]
        diff = abs(stored - recalc)
        match = "✅" if diff < 1e-10 else "❌"
        
        print(f"{feature:<15} {stored:<15.10f} {recalc:<15.10f} {match:<10}")
        if diff >= 1e-10:
            print(f"{'':>15} Difference: {diff:.2e}")

# ---------- MAIN EXECUTION ----------
if __name__ == "__main__":
    print("🚀 Feature Validation Tool")
    print("="*50)
    
    # Option 1: Quick check for one file
    print("\n1️⃣ Quick check (first file of patient 001):")
    quick_check_sample()
    
    print("\n" + "="*50)
    
    # Option 2: Full validation
    print("2️⃣ Full validation of all records:")
    user_input = input("Run full validation? (y/n): ").lower().strip()
    
    if user_input == 'y':
        validation_results = validate_features()
    else:
        print("Skipped full validation.")yield