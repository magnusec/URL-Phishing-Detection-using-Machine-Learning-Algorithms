import pandas as pd

# Load dataset
df = pd.read_csv("data/Phishing_Legitimate_full.csv")

# Basic info
print("Shape of dataset:", df.shape)
print("\nColumn names:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

print("\nLabel distribution:")
print(df.iloc[:, -1].value_counts())

