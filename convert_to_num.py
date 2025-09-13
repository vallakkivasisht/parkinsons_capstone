import pandas as pd

# Load your merged CSV
file_path = r"C:\Users\vallakki Vasisht\OneDrive\Documents\capstone\working_model\patients_features_merged_with_fixed_conditions.csv"
df = pd.read_csv(file_path)

# Extract activity name from File column (remove PatientID prefix and .csv suffix)
df["Activity"] = df["File"].str.replace(r"^\d+_", "", regex=True).str.replace(".csv", "", regex=False)

# Factorize activity labels into integers (0, 1, 2, ...)
df["Activity_Code"] = pd.factorize(df["Activity"])[0]

# One-hot encode into exactly N generic columns (ACTIVITY_0, ACTIVITY_1, ACTIVITY_2)
activity_onehot = pd.get_dummies(df["Activity_Code"], prefix="ACTIVITY").astype(int)

# Append the new one-hot columns to the existing dataset
df = pd.concat([df, activity_onehot], axis=1)


# Save the updated dataset
output_file = file_path.replace(".csv", "_with_onehot_new.csv")
df.to_csv(output_file, index=False)

print("✅ One-hot columns added at the end of dataset")
print("Saved to:", output_file)
print(df.head())
