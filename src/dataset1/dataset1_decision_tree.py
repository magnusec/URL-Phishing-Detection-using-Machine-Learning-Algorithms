import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# Load dataset
# =========================
df = pd.read_csv("data/Phishing_Legitimate_full.csv")

# =========================
# DROP ID COLUMN (DATA LEAKAGE FIX)
# =========================
if "id" in df.columns:
    df = df.drop(columns=["id"])

# =========================
# Separate features & label
# =========================
X = df.drop(columns=["CLASS_LABEL"])
y = df["CLASS_LABEL"]

# =========================
# Train-test split (80-20)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# Decision Tree (CONTROLLED)
# =========================
dt_model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

# =========================
# Train model
# =========================
dt_model.fit(X_train, y_train)

# =========================
# Predictions
# =========================
y_pred = dt_model.predict(X_test)

# =========================
# Evaluation
# =========================
accuracy = accuracy_score(y_test, y_pred)

print("Decision Tree Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
