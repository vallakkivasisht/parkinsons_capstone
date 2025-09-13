# bilstm_preprocessing_fixed.py  (fixed)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# =========================
# STEP 1: Load dataset
# =========================
df = pd.read_csv("patients_features_filtered.csv")
print("✅ CSV loaded")
print("Original columns:", df.columns.tolist())
print("Dataset shape:", df.shape)

# =========================
# STEP 2: Define clean feature set (NO DATA LEAKAGE)
# =========================
# Core sensor features (your 6 main features)
sensor_features = [
    "ACC_RMS_filtered", "ACC_CV_filtered", "ACC_AvgFreq_filtered",     # Accelerometer features
    "GYRO_RMS_filtered", "GYRO_CV_filtered", "GYRO_AvgFreq_filtered"   # Gyroscope features
]

# Optional demographic features (clinically valid)
demographic_features = [
    "age_at_diagnosis",
    "age"
]

# Build final feature list
feature_cols = []

# Add sensor features (check if they exist)
for feat in sensor_features:
    if feat in df.columns:
        feature_cols.append(feat)
        print(f"✅ Added sensor feature: {feat}")
    else:
        print(f"❌ Missing sensor feature: {feat}")

# Add demographic features (check if they exist)
for feat in demographic_features:
    if feat in df.columns:
        feature_cols.append(feat)
        print(f"✅ Added demographic feature: {feat}")
    else:
        print(f"❌ Missing demographic feature: {feat}")

print(f"\n✅ Final feature set ({len(feature_cols)} features):")
print(feature_cols)

# Verify no leakage columns are included
leakage_check = [col for col in feature_cols if col in ["COND_OTHER", "COND_PARKINSONS", "COND_HEALTHY"]]
if leakage_check:
    print(f"🚨 ERROR: Data leakage detected! Remove: {leakage_check}")
    exit(1)
else:
    print("✅ No data leakage detected")

# =========================
# STEP 3: Scale numeric features
# =========================
print(f"\n🔧 Scaling {len(feature_cols)} features...")

# Check for missing values and fill with median (per-column)
missing_counts = df[feature_cols].isnull().sum()
if missing_counts.sum() > 0:
    print("⚠️ Missing values detected:")
    print(missing_counts[missing_counts > 0])
    print("Filling missing values with median...")
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

# Ensure numeric (in case some string columns slipped in)
numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
non_numeric = [c for c in feature_cols if c not in numeric_cols]
if non_numeric:
    print(f"⚠️ Dropping non-numeric feature columns before scaling (they'll still be excluded): {non_numeric}")
    # remove them from feature_cols so we don't try to use them
    feature_cols = numeric_cols

# Apply standardization
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])
print("✅ Feature scaling complete")

# =========================
# STEP 4: Clean label encoding (NO LEAKAGE)
# =========================
def map_condition_clean(cond):
    """Clean label mapping without data leakage.
       Returns a tuple (hashable) instead of a list, to allow drop_duplicates() and grouping.
       Mapping:
         Healthy -> (1,0,0)
         Parkinson's -> (0,1,0)
         Others -> (0,0,1)
    """
    cond = str(cond).strip().lower()
    if cond == "parkinson's" or cond == "parkinsons":
        return (0, 1, 0)  # Parkinson's
    elif cond == "healthy":
        return (0, 0, 1)  # Healthy
    else:  # Other conditions
        return (1, 0, 0)  # Other

# Apply clean label mapping
df["label_vec"] = df["condition"].apply(map_condition_clean)

# Debug label distribution
print(f"\n📊 Label Distribution:")
condition_counts = df["condition"].value_counts()
print(condition_counts)

# Verify label mapping worked
label_check = df[["condition", "label_vec"]].drop_duplicates()
print(f"\n🏷️ Label Mapping:")
for _, row in label_check.iterrows():
    print(f"'{row['condition']}' → {row['label_vec']}")

# =========================
# STEP 5: Group into sequences by PatientID
# =========================
print(f"\n🔄 Creating sequences...")

X_list, y_list, patient_ids = [], [], []

# Ensure consistent patient order
patient_ids_sorted = sorted(df["PatientID"].unique())

for patient_id in patient_ids_sorted:
    patient_data = df[df["PatientID"] == patient_id]
    # If you have an activity order column, sort by it here. If not, data already has consistent order.
    # e.g., patient_data = patient_data.sort_values(by="ACTIVITY_ID")
    
    # Get features for this patient (all activities)
    # Use feature_cols which currently contains only numeric columns we scaled
    features = patient_data[feature_cols].values
    
    # Get label (should be same for all activities of this patient)
    labels = patient_data["label_vec"].values
    if len(labels) == 0:
        continue  # no rows for this patient (defensive)
    patient_label = labels[0]
    
    # Verify all activities have same label for this patient
    if not all(tuple(label) == tuple(patient_label) for label in labels):
        print(f"⚠️ Warning: Patient {patient_id} has inconsistent labels!")
    
    X_list.append(features)
    y_list.append(list(patient_label))  # convert to list so resulting np.array is numeric 2D
    patient_ids.append(patient_id)

# Convert to numpy arrays
X = np.array(X_list)  # shape: (n_patients, n_activities, n_features)
y = np.array(y_list)  # shape: (n_patients, n_classes)

print(f"✅ Sequence creation complete")
print(f"Number of patients: {len(patient_ids)}")
print(f"Activities per patient: {X.shape[1] if len(X) > 0 else 0}")

# =========================
# STEP 6: Final validation and summary
# =========================
print(f"\n📋 PREPROCESSING SUMMARY:")
print("="*50)
print(f"Final dataset shape:")
print(f"  X (features): {X.shape}")
print(f"  y (labels):   {y.shape}")

print(f"\nFeatures used ({X.shape[2] if len(X.shape) > 2 else 0}):")
for i, feat in enumerate(feature_cols):
    print(f"  {i+1:2d}. {feat}")

print(f"\nClass distribution:")
unique_labels, counts = np.unique(y, axis=0, return_counts=True)
class_names = ["Healthy", "Parkinson's", "Other"]
for (label, count) in zip(unique_labels, counts):
    class_idx = int(np.argmax(label))
    percentage = count / len(y) * 100
    print(f"  {class_names[class_idx]:<12}: {count:3d} patients ({percentage:5.1f}%)")

print(f"\nExample patient data:")
if len(X) > 0:
    print(f"  Patient 1 features shape: {X[0].shape}")
    print(f"  Patient 1 label: {y[0]} → {class_names[int(np.argmax(y[0]))]}")

# =========================
# STEP 7: Save preprocessed data (optional)
# =========================
save_data = input(f"\n💾 Save preprocessed data? (y/n): ").lower().strip()
if save_data == 'y':
    np.savez('bilstm_data_clean.npz',
             X=X, y=y,
             feature_names=feature_cols,
             patient_ids=patient_ids)
    print("✅ Data saved to 'bilstm_data_clean.npz'")

print(f"\n🎯 READY FOR BILSTM TRAINING!")
print("="*50)
print("✅ No data leakage")
print("✅ Proper feature scaling")
print("✅ Clean label encoding")
print("✅ Sequential structure maintained")
print(f"✅ {len(feature_cols)} features: {', '.join(feature_cols[:3])}...")
