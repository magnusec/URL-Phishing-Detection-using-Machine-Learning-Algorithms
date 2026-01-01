import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/Phishing_Legitimate_full.csv")

# Separate features and label
X = df.drop(columns=["CLASS_LABEL"])
y = df["CLASS_LABEL"]

# Train-test split (80-20 as used in paper)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nTesting label distribution:")
print(y_test.value_counts())



