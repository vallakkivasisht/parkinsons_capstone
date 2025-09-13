import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import rfft, rfftfreq
import os

def validate_filtering_results():
    """
    Comprehensive validation of your bandpass filtering results
    """
    print("🔍 FILTER VALIDATION TOOL")
    print("="*50)
    
    # Check if filtered data exists
    filtered_file = "patients_features_filtered.csv"
    if not os.path.exists(filtered_file):
        print(f"❌ Filtered data file not found: {filtered_file}")
        print("Run the filtering code first!")
        return
    
    # Load filtered results
    df = pd.read_csv(filtered_file)
    print(f"✅ Loaded {len(df)} records from filtered dataset")
    
    # Basic validation checks
    validate_basic_properties(df)
    
    # Statistical validation  
    validate_statistical_properties(df)
    
    # Visual validation (if user wants)
    create_validation_plots = input("\n📊 Create validation plots? (y/n): ").lower().strip()
    if create_validation_plots == 'y':
        validate_with_plots(df)
    
    # Filter response validation
    validate_filter_response()
    
    print("\n🎯 VALIDATION COMPLETE!")

def validate_basic_properties(df):
    """
    Basic sanity checks on filtered data
    """
    print("\n1️⃣ BASIC VALIDATION CHECKS:")
    print("-" * 30)
    
    # Check if filtered columns exist
    filtered_cols = [col for col in df.columns if 'filtered' in col]
    raw_cols = [col for col in df.columns if 'raw' in col]
    
    print(f"✅ Filtered features found: {len(filtered_cols)}")
    print(f"✅ Raw features found: {len(raw_cols)}")
    
    # Check for NaN or infinite values
    for col in filtered_cols + raw_cols:
        nan_count = df[col].isna().sum()
        inf_count = np.isinf(df[col]).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"⚠️ {col}: {nan_count} NaN, {inf_count} Inf values")
        else:
            print(f"✅ {col}: Clean data")
    
    # Check value ranges (should be reasonable)
    print(f"\n📊 Value Ranges:")
    key_features = ['ACC_RMS_filtered', 'GYRO_RMS_filtered', 'ACC_AvgFreq_filtered', 'GYRO_AvgFreq_filtered']
    for feat in key_features:
        if feat in df.columns:
            min_val = df[feat].min()
            max_val = df[feat].max()
            mean_val = df[feat].mean()
            print(f"   {feat:<20}: {min_val:.4f} to {max_val:.4f} (mean: {mean_val:.4f})")

def validate_statistical_properties(df):
    """
    Statistical validation of filtering effects
    """
    print("\n2️⃣ STATISTICAL VALIDATION:")
    print("-" * 30)
    
    # Expected effects of bandpass filtering
    comparisons = [
        ('ACC_RMS_raw', 'ACC_RMS_filtered'),
        ('GYRO_RMS_raw', 'GYRO_RMS_filtered'), 
        ('ACC_AvgFreq_raw', 'ACC_AvgFreq_filtered'),
        ('GYRO_AvgFreq_raw', 'GYRO_AvgFreq_filtered')
    ]
    
    for raw_col, filtered_col in comparisons:
        if raw_col in df.columns and filtered_col in df.columns:
            raw_vals = df[raw_col]
            filtered_vals = df[filtered_col]
            
            # Calculate statistics
            raw_mean = raw_vals.mean()
            filtered_mean = filtered_vals.mean()
            change_pct = ((filtered_mean - raw_mean) / raw_mean * 100) if raw_mean != 0 else 0
            
            # Check correlation (should be high - filtered signal should correlate with raw)
            correlation = np.corrcoef(raw_vals, filtered_vals)[0, 1]
            
            # Check standard deviation changes
            raw_std = raw_vals.std()
            filtered_std = filtered_vals.std()
            std_change = ((filtered_std - raw_std) / raw_std * 100) if raw_std != 0 else 0
            
            print(f"\n{raw_col.replace('_raw', '')}:")
            print(f"   Mean change: {change_pct:+.1f}%")
            print(f"   Std change:  {std_change:+.1f}%")
            print(f"   Correlation: {correlation:.3f} {'✅' if correlation > 0.7 else '⚠️'}")
            
            # Interpretation
            if 'RMS' in raw_col:
                if abs(change_pct) < 50:  # RMS shouldn't change dramatically
                    print("   ✅ RMS change is reasonable (noise removal)")
                else:
                    print("   ⚠️ Large RMS change - check filtering parameters")
                    
            elif 'AvgFreq' in raw_col:
                if 0 < filtered_mean < 25:  # Should be in physiological range
                    print("   ✅ Frequency is in physiological range")
                else:
                    print("   ⚠️ Frequency outside expected range")

def validate_with_plots(df):
    """
    Create validation plots to visualize filtering effects
    """
    print("\n3️⃣ CREATING VALIDATION PLOTS...")
    print("-" * 30)
    
    # Sample a few random records for detailed analysis
    sample_indices = np.random.choice(len(df), min(3, len(df)), replace=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Filter Validation: Raw vs Filtered Features', fontsize=16)
    
    # Plot 1: RMS comparison
    axes[0, 0].scatter(df['ACC_RMS_raw'], df['ACC_RMS_filtered'], alpha=0.6, s=20)
    axes[0, 0].plot([df['ACC_RMS_raw'].min(), df['ACC_RMS_raw'].max()], 
                    [df['ACC_RMS_raw'].min(), df['ACC_RMS_raw'].max()], 'r--', alpha=0.8)
    axes[0, 0].set_xlabel('ACC RMS Raw')
    axes[0, 0].set_ylabel('ACC RMS Filtered')
    axes[0, 0].set_title('Accelerometer RMS: Raw vs Filtered')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Frequency comparison  
    axes[0, 1].scatter(df['ACC_AvgFreq_raw'], df['ACC_AvgFreq_filtered'], alpha=0.6, s=20)
    axes[0, 1].plot([df['ACC_AvgFreq_raw'].min(), df['ACC_AvgFreq_raw'].max()],
                    [df['ACC_AvgFreq_raw'].min(), df['ACC_AvgFreq_raw'].max()], 'r--', alpha=0.8)
    axes[0, 1].set_xlabel('ACC AvgFreq Raw') 
    axes[0, 1].set_ylabel('ACC AvgFreq Filtered')
    axes[0, 1].set_title('Accelerometer Frequency: Raw vs Filtered')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Gyro RMS
    axes[1, 0].scatter(df['GYRO_RMS_raw'], df['GYRO_RMS_filtered'], alpha=0.6, s=20)
    axes[1, 0].plot([df['GYRO_RMS_raw'].min(), df['GYRO_RMS_raw'].max()],
                    [df['GYRO_RMS_raw'].min(), df['GYRO_RMS_raw'].max()], 'r--', alpha=0.8)
    axes[1, 0].set_xlabel('GYRO RMS Raw')
    axes[1, 0].set_ylabel('GYRO RMS Filtered') 
    axes[1, 0].set_title('Gyroscope RMS: Raw vs Filtered')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Distribution comparison
    feature = 'ACC_AvgFreq'
    axes[1, 1].hist(df[f'{feature}_raw'], bins=30, alpha=0.7, label='Raw', density=True)
    axes[1, 1].hist(df[f'{feature}_filtered'], bins=30, alpha=0.7, label='Filtered', density=True)
    axes[1, 1].set_xlabel('Average Frequency (Hz)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Frequency Distribution: Raw vs Filtered')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('filter_validation_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Plots saved as 'filter_validation_plots.png'")

def validate_filter_response():
    """
    Test the filter response with known signals
    """
    print("\n4️⃣ FILTER RESPONSE VALIDATION:")
    print("-" * 30)
    
    # Import your filtering function
    from scipy import signal as sp_signal
    
    def bandpass_filter(data, low_freq=0.5, high_freq=15, sampling_rate=50, order=4):
        """Your exact filter function"""
        nyquist = sampling_rate / 2
        low_norm = max(0.001, min(low_freq / nyquist, 0.999))
        high_norm = max(low_norm + 0.001, min(high_freq / nyquist, 0.999))
        
        try:
            b, a = sp_signal.butter(order, [low_norm, high_norm], btype='band', analog=False)
            filtered_data = sp_signal.filtfilt(b, a, data)
            return filtered_data
        except Exception as e:
            return data
    
    # Test with known frequencies
    fs = 50
    t = np.linspace(0, 4, 4*fs)  # 4 seconds
    
    # Test signals
    test_cases = [
        ("DC component (0 Hz)", np.ones(len(t)) * 5),
        ("Very low freq (0.1 Hz)", np.sin(2*np.pi*0.1*t)),
        ("Good freq (2 Hz)", np.sin(2*np.pi*2*t)),
        ("Tremor freq (5 Hz)", np.sin(2*np.pi*5*t)), 
        ("High freq (25 Hz)", np.sin(2*np.pi*25*t)),
        ("Mixed signal", np.sin(2*np.pi*2*t) + 0.5*np.sin(2*np.pi*5*t) + 0.3*np.sin(2*np.pi*25*t))
    ]
    
    for name, test_signal in test_cases:
        # Apply your filter
        filtered_signal = bandpass_filter(test_signal, low_freq=0.5, high_freq=15, sampling_rate=fs)
        
        # Calculate power retention
        original_power = np.mean(test_signal**2)
        filtered_power = np.mean(filtered_signal**2)
        retention = (filtered_power / original_power * 100) if original_power > 0 else 0
        
        print(f"{name:<25}: {retention:5.1f}% power retained", end="")
        
        # Expected results
        if "0 Hz" in name or "0.1 Hz" in name:
            status = "✅ Removed" if retention < 10 else "⚠️ Should be removed"
        elif "2 Hz" in name or "5 Hz" in name or "Mixed" in name:
            status = "✅ Preserved" if retention > 50 else "⚠️ Should be preserved"
        elif "25 Hz" in name:
            status = "✅ Removed" if retention < 30 else "⚠️ Should be removed"
        else:
            status = ""
            
        print(f" {status}")

def create_test_signal_analysis():
    """
    Create a comprehensive test with synthetic signals
    """
    print("\n5️⃣ SYNTHETIC SIGNAL TEST:")
    print("-" * 30)
    
    create_test = input("Create detailed synthetic signal analysis? (y/n): ").lower().strip()
    if create_test != 'y':
        return
    
    # Create realistic IMU-like test signal
    fs = 50
    duration = 10  # 10 seconds
    t = np.linspace(0, duration, duration*fs)
    
    # Realistic IMU signal components
    gravity = 9.8  # DC component (should be removed)
    slow_drift = 0.5 * np.sin(2*np.pi*0.05*t)  # Very slow drift (should be removed)
    normal_movement = 2.0 * np.sin(2*np.pi*1.5*t)  # Normal movement (should be kept)
    tremor = 1.0 * np.sin(2*np.pi*4.5*t)  # Tremor frequency (should be kept)
    noise = 0.3 * np.random.randn(len(t))  # High frequency noise
    high_freq_artifact = 0.5 * np.sin(2*np.pi*30*t)  # High freq artifact (should be removed)
    
    # Combine all components
    synthetic_signal = gravity + slow_drift + normal_movement + tremor + noise + high_freq_artifact
    
    print("✅ Created synthetic IMU signal with:")
    print("   - DC component (gravity): 9.8")
    print("   - Slow drift: 0.05 Hz") 
    print("   - Normal movement: 1.5 Hz")
    print("   - Tremor: 4.5 Hz")
    print("   - High frequency artifact: 30 Hz")
    print("   - Random noise")
    
    # Apply your filter
    from scipy import signal as sp_signal
    def bandpass_filter(data, low_freq=0.5, high_freq=15, sampling_rate=50, order=4):
        nyquist = sampling_rate / 2
        low_norm = max(0.001, min(low_freq / nyquist, 0.999))
        high_norm = max(low_norm + 0.001, min(high_freq / nyquist, 0.999))
        b, a = sp_signal.butter(order, [low_norm, high_norm], btype='band', analog=False)
        return sp_signal.filtfilt(b, a, data)
    
    filtered_signal = bandpass_filter(synthetic_signal)
    
    # Create comparison plot
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(t[:200], synthetic_signal[:200], 'b-', alpha=0.8, label='Original')
    plt.plot(t[:200], filtered_signal[:200], 'r-', alpha=0.8, label='Filtered')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Time Domain: Original vs Filtered (first 4 seconds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Frequency domain analysis
    freqs_orig, psd_orig = sp_signal.periodogram(synthetic_signal, fs)
    freqs_filt, psd_filt = sp_signal.periodogram(filtered_signal, fs)
    
    plt.subplot(2, 2, 2)
    plt.semilogy(freqs_orig, psd_orig, 'b-', alpha=0.8, label='Original')
    plt.semilogy(freqs_filt, psd_filt, 'r-', alpha=0.8, label='Filtered')
    plt.axvline(x=0.5, color='g', linestyle='--', alpha=0.7, label='Filter cutoffs')
    plt.axvline(x=15, color='g', linestyle='--', alpha=0.7)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title('Frequency Domain Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 25)
    
    # Component analysis
    components = {
        'DC (0 Hz)': gravity,
        'Slow drift (0.05 Hz)': slow_drift, 
        'Movement (1.5 Hz)': normal_movement,
        'Tremor (4.5 Hz)': tremor,
        'Artifact (30 Hz)': high_freq_artifact
    }
    
    plt.subplot(2, 2, 3)
    retention_rates = []
    component_names = []
    
    for name, component in components.items():
        if np.std(component) > 0:  # Skip DC for this analysis
            filtered_component = bandpass_filter(component)
            retention = np.std(filtered_component) / np.std(component) * 100
            retention_rates.append(retention)
            component_names.append(name.split('(')[0])
    
    bars = plt.bar(component_names, retention_rates)
    plt.ylabel('Power Retention (%)')
    plt.title('Component Retention After Filtering')
    plt.xticks(rotation=45)
    
    # Color bars based on expected behavior
    for i, (bar, rate) in enumerate(zip(bars, retention_rates)):
        if 'Movement' in component_names[i] or 'Tremor' in component_names[i]:
            bar.set_color('green' if rate > 70 else 'orange')
        else:
            bar.set_color('red' if rate < 30 else 'orange')
    
    plt.grid(True, alpha=0.3)
    
    # Summary statistics
    plt.subplot(2, 2, 4)
    metrics = ['Mean', 'RMS', 'Std', 'Peak-to-Peak']
    original_metrics = [np.mean(synthetic_signal), np.sqrt(np.mean(synthetic_signal**2)), 
                       np.std(synthetic_signal), np.ptp(synthetic_signal)]
    filtered_metrics = [np.mean(filtered_signal), np.sqrt(np.mean(filtered_signal**2)),
                       np.std(filtered_signal), np.ptp(filtered_signal)]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, original_metrics, width, label='Original', alpha=0.8)
    plt.bar(x + width/2, filtered_metrics, width, label='Filtered', alpha=0.8)
    plt.ylabel('Value')
    plt.title('Statistical Metrics Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('synthetic_signal_validation.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Synthetic signal analysis saved as 'synthetic_signal_validation.png'")

if __name__ == "__main__":
    validate_filtering_results()
    create_test_signal_analysis()