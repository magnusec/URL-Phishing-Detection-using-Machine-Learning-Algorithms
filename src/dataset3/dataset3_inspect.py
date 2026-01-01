import pandas as pd

# Load dataset 3
df = pd.read_csv("data/web-page-phishing.csv")

print("\nColumn Names:\n")
print(df.columns)

print("\nFirst 5 rows:\n")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nData types:\n")
print(df.dtypes)

print("\nLabel distribution (trying common names):")
for col in df.columns:
    if col.lower() in ["label", "class", "result", "target", "phishing"]:
        print(f"\n{col} value counts:")
        print(df[col].value_counts())
