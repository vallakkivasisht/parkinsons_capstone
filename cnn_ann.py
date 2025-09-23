import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# ---------- LOAD DATA ----------
df = pd.read_csv("patients_top10_features.csv")

# Assume the CSV already contains aggregated features, so no feature engineering needed.
# Drop 'PatientID' and other non-feature columns.
X = df.select_dtypes(include=[np.number]).drop(columns=["PatientID"], errors="ignore")
y = df[["COND_HEALTHY", "COND_PARKINSONS", "COND_OTHER"]]

print("✅ Features loaded successfully.")

# ---------- DATA PREPARATION ----------

# Convert one-hot encoded labels to single labels for SMOTE
y_labels = y.idxmax(axis=1)

# Handle missing values in X
if X.isnull().values.any():
    X = X.fillna(X.mean())
    print("✅ Handled missing values (NaN) in X by mean imputation.")

# Balance the data using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y_labels)

print("Before SMOTE:", y_labels.value_counts().to_dict())
print("After SMOTE:", y_res.value_counts().to_dict())

# Convert the resampled labels back to one-hot encoding for the ANN model
y_res = pd.get_dummies(y_res)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------- ANN MODEL ----------
ann = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(y.shape[1], activation='softmax')
])

# Compile and train the model
ann.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
ann.fit(X_train, y_train, validation_data=(X_test, y_test),
        epochs=100, batch_size=32, verbose=1)

# ---------- EVALUATION ----------
print("\nANN Accuracy:", ann.evaluate(X_test, y_test, verbose=0)[1])