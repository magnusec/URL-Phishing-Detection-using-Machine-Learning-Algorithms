import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("data/Phishing_Legitimate_full.csv")

# Drop ID column
if "id" in df.columns:
    df = df.drop(columns=["id"])

# Features & label
X = df.drop(columns=["CLASS_LABEL"])
y = df["CLASS_LABEL"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# SVM Model
svm_model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale"
)

# Train
svm_model.fit(X_train, y_train)

# Predict
y_pred = svm_model.predict(X_test)

# Evaluate
print("SVM Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
