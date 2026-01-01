import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset 3
df = pd.read_csv("data/web-page-phishing.csv")

# Features and label
X = df.drop(columns=["phishing"])
y = df["phishing"]

# Train-test split (80-20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nTesting label distribution:")
print(y_test.value_counts())
