import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# Load dataset
# =========================
df = pd.read_csv("data/Phishing_Legitimate_full.csv")

# =========================
# Drop ID column (data leakage prevention)
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
# Random Forest Model
# =========================
rf_model = RandomForestClassifier(
    n_estimators=100,        # number of trees
    max_depth=20,            # controls overfitting
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1                # use all CPU cores
)

# =========================
# Train model
# =========================
rf_model.fit(X_train, y_train)

# =========================
# Predictions
# =========================
y_pred = rf_model.predict(X_test)

# =========================
# Evaluation
# =========================
accuracy = accuracy_score(y_test, y_pred)

print("Random Forest Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
