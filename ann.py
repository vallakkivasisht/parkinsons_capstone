"""
train_ann.py
Dense ANN for tabular data:
- features: first 10 columns + any columns containing 'age' (case-insensitive)
- targets: last 3 columns (one-hot)
Outputs test loss, test accuracy, F1 (macro) and classification report.
Saves model to /mnt/data/ann_tabular_model.h5
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

CSV_PATH = "patients_focused15_features.csv"
assert os.path.exists(CSV_PATH), f"CSV not found at {CSV_PATH}"

# Load
df = pd.read_csv(CSV_PATH)
print("Loaded CSV shape:", df.shape)
print("Columns:", list(df.columns))

# Feature selection: first 10 columns + 'age' columns (case-insensitive)
first10 = list(df.columns[:10])
age_cols = [c for c in df.columns if 'age' in c.lower() and c not in first10]
feature_cols = first10 + age_cols
print("Selected feature columns:", feature_cols)

# Targets: last 3 columns assumed one-hot
target_cols = list(df.columns[-3:])
print("Selected target columns (last 3):", target_cols)

X = df[feature_cols].values
y = df[target_cols].values

# Handle missing values (simple strategy)
if np.isnan(X).any():
    X = pd.DataFrame(X, columns=feature_cols).fillna(pd.DataFrame(X, columns=feature_cols).mean()).values
if np.isnan(y).any():
    y = np.nan_to_num(y)

# Train/test split (stratify if possible)
if y.shape[1] > 1:
    stratify_labels = np.argmax(y, axis=1)
else:
    stratify_labels = None

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_labels)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build ANN
num_features = X_train.shape[1]
num_classes = y_train.shape[1] if y_train.ndim > 1 else 1

model = keras.Sequential([
    layers.Input(shape=(num_features,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax' if num_classes > 1 else 'sigmoid')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy' if num_classes > 1 else 'binary_crossentropy',
    metrics=['accuracy']
)
model.summary()

# Train with early stopping
es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_split=0.1, epochs=50, batch_size=32, callbacks=[es], verbose=2)

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest loss: {loss:.4f}")
print(f"Test accuracy: {acc:.4f}")

# Predictions -> F1
y_pred_prob = model.predict(X_test)
if num_classes > 1:
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred_prob, axis=1)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    print(f"F1 score (macro): {f1_macro:.4f}\n")
    print("Classification report:\n", classification_report(y_true, y_pred))
else:
    y_true = y_test.ravel().astype(int)
    y_pred = (y_pred_prob.ravel() > 0.5).astype(int)
    f1 = f1_score(y_true, y_pred)
    print(f"F1 score: {f1:.4f}\n")
    print("Classification report:\n", classification_report(y_true, y_pred))

# Save model and scaler
out_model = "/mnt/data/ann_tabular_model.h5"
model.save(out_model)
print("Saved model to", out_model)

# Save scaler for inference
import joblib
joblib.dump(scaler, "/mnt/data/scaler.save")
print("Saved scaler to /mnt/data/scaler.save")
