import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset 2
df = pd.read_csv("data/phishing_dataset.csv")

# Drop useless column
if "index" in df.columns:
    df = df.drop(columns=["index"])

# Convert labels
df["Result"] = df["Result"].map({-1: 1, 1: 0})

# Split
X = df.drop(columns=["Result"])
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Naive Bayes model
nb_model = GaussianNB()

# Train
nb_model.fit(X_train, y_train)

# Predict
y_pred = nb_model.predict(X_test)

# Evaluate
print("Naive Bayes Accuracy (Dataset 2):", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
