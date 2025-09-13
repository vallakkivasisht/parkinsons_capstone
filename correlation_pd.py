import pandas as pd

# Load your dataset
file_path = r"C:\Users\vallakki Vasisht\OneDrive\Documents\capstone\working_model\patients_features_filtered.csv"
df = pd.read_csv(file_path)

# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=["number"])

# Define your target condition columns
target_cols = ["COND_OTHER", "COND_PARKINSONS", "COND_HEALTHY"]

# Compute correlation with each condition
corr = numeric_df.corr()[target_cols]

# Save results
output_file = file_path.replace(".csv", "_correlation.csv")
corr.to_csv(output_file)

print("✅ Correlation matrix created!")
print("Saved to:", output_file)
print(corr.head(15))  # Show first 15 correlations
