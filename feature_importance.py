import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Load the dataset
df = pd.read_csv("patients_features_filtered.csv")

# 2. Define target columns (adjust if names differ)
target_cols = ["COND_OTHER", "COND_PARKINSONS", "COND_HEALTHY"]

# 3. Convert one-hot encoding to a single target column
df["target"] = df[target_cols].idxmax(axis=1)

# 4. Drop target columns & keep only numeric features
X = df.drop(columns=target_cols + ["target"])
X = X.select_dtypes(include=['number'])

y = df["target"]

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Train model
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# 7. Feature importances
importances = pd.DataFrame({
    "feature": X.columns,
    "importance": clf.feature_importances_
}).sort_values(by="importance", ascending=False)

print(importances.head(20))

# 8. Plot
plt.figure(figsize=(10,6))
plt.barh(importances["feature"].head(20), importances["importance"].head(20))
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 20 Feature Importances")
plt.show()
