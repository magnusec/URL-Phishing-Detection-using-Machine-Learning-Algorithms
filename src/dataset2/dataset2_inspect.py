import pandas as pd

# Load Dataset-2
df = pd.read_csv("data/phishing_dataset.csv")

print("Dataset-2 Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)
