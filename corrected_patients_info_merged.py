import pandas as pd

# Load your dataset
file_path = r"C:\Users\vallakki Vasisht\OneDrive\Documents\capstone\working_model\patients_features_merged.csv"
df = pd.read_csv(file_path)

# Map the condition column (column k) into your custom one-hot encoding
mapping = {
    "Parkinson's": [1, 0, 0],
    "Other Movement Disorders": [0, 1, 0],
    "Healthy": [0, 0, 1]
}

# Apply the mapping
for idx, cond in df["condition"].items():
    if cond in mapping:
        df.at[idx, "COND_PARKINSONS"] = mapping[cond][0]
        df.at[idx, "COND_OTHER"] = mapping[cond][1]
        df.at[idx, "COND_HEALTHY"] = mapping[cond][2]

# Save the updated dataset
output_file = file_path.replace(".csv", "_with_fixed_conditions.csv")
df.to_csv(output_file, index=False)

print("✅ Columns q, r, s updated based on 'condition'")
print("Saved to:", output_file)
print(df[["condition", "COND_PARKINSONS", "COND_OTHER", "COND_HEALTHY"]].head())
