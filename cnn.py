"""
train_cnn.py
1D-CNN for tabular data: reshape features to (n_features, 1) and apply Conv1D.
Same selection rules as ANN above. Prints loss, accuracy and F1.
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

df = pd.read_csv(CSV_PATH)
first10 = list(df.columns[:10])
age_cols = [c for c in df.columns if 'age' in c.lower() and c not in first10]
feature_cols = first10 + age_cols
target_cols = list(df.columns[-3:])

X = df[feature_cols].values
y = df[target_cols].values

if np.isnan(X).any():
    X = pd.DataFrame(X, columns=feature_cols).fillna(pd.DataFrame(X, columns=feature_cols).mean()).values
if np.isnan(y).any():
    y = np.nan_to_num(y)

stratify_labels = np.argmax(y, axis=1) if y.shape[1] > 1 else None
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_labels)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# reshape for Conv1D: (samples, timesteps=num_features, channels=1)
n_features = X_train.shape[1]
X_train_c = X_train.reshape(-1, n_features, 1)
X_test_c = X_test.reshape(-1, n_features, 1)

num_classes = y_train.shape[1] if y_train.ndim > 1 else 1

inp = keras.Input(shape=(n_features, 1))
x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(inp)
x = layers.MaxPool1D(2)(x)
x = layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(num_classes, activation='softmax' if num_classes>1 else 'sigmoid')(x)

model = keras.Model(inp, out)
model.compile(optimizer='adam',
              loss='categorical_crossentropy' if num_classes>1 else 'binary_crossentropy',
              metrics=['accuracy'])
model.summary()

es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
history = model.fit(X_train_c, y_train, validation_split=0.1, epochs=50, batch_size=32, callbacks=[es], verbose=2)

loss, acc = model.evaluate(X_test_c, y_test, verbose=0)
print(f"\nTest loss: {loss:.4f}")
print(f"Test accuracy: {acc:.4f}")

y_pred_prob = model.predict(X_test_c)
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

model.save("/mnt/data/cnn_tabular_model.h5")
print("Saved CNN model to /mnt/data/cnn_tabular_model.h5")
