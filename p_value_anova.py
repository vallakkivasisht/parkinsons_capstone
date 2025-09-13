import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("patients_features_filtered.csv")

# --- STEP 1: Drop unwanted empty columns ---
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# --- STEP 2: Define target columns ---
target_cols = ["COND_OTHER", "COND_PARKINSONS", "COND_HEALTHY"]
df["target"] = df[target_cols].idxmax(axis=1)

# --- STEP 3: Select numeric features ---
X = df.drop(columns=target_cols + ["target"])
X = X.select_dtypes(include=['number'])

y = df["target"]

# --- STEP 4: Handle missing values ---
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --- STEP 5: Run ANOVA F-test ---
F, p = f_classif(X_imputed, y_encoded)

# --- STEP 6: Collect results ---
anova_results = pd.DataFrame({
    "feature": X.columns,
    "F_value": F,
    "p_value": p
}).sort_values(by="p_value")

print(anova_results.head(20))
print("\nFeatures with p < 0.05 (statistically significant):")
print(anova_results[anova_results["p_value"] < 0.05])

# --- STEP 7: Save results to CSV ---
anova_results.to_csv("anova_results.csv", index=False)
