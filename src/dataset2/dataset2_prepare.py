import pandas as pd

# Load dataset 2
df = pd.read_csv("data/phishing_dataset.csv")

# Drop index column (not a feature)
if "index" in df.columns:
    df = df.drop(columns=["index"])

# Convert label:
# -1 -> 1 (phishing)
#  1 -> 0 (legitimate)
df["Result"] = df["Result"].map({-1: 1, 1: 0})

# Check
print("Dataset shape:", df.shape)
print("\nLabel distribution:")
print(df["Result"].value_counts())

# Save cleaned dataset
df.to_csv("data/phishing_dataset_cleaned.csv", index=False)

print("\nSaved as phishing_dataset_cleaned.csv")
