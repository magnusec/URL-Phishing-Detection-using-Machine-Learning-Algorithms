import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# =========================
# Load Dataset 1
# =========================
df = pd.read_csv("data/Phishing_Legitimate_full.csv")

# Remove ID column (data leakage fix)
if "id" in df.columns:
    df = df.drop(columns=["id"])

# Features and label
X = df.drop(columns=["CLASS_LABEL"])
y = df["CLASS_LABEL"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# Train FINAL Random Forest
# =========================
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# =========================
# Save model
# =========================
joblib.dump(rf_model, "rf_dataset1.pkl")

print("✅ Dataset 1 Random Forest model saved as rf_dataset1.pkl")
