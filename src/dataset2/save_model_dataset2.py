import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# =========================
# Load DATASET 2
# =========================
df = pd.read_csv("data/phishing_dataset.csv")

# =========================
# Drop index column (not a feature)
# =========================
if "index" in df.columns:
    df = df.drop(columns=["index"])

# =========================
# Separate features and label
# LABEL COLUMN IS 'Result'
# =========================
X = df.drop(columns=["Result"])
y = df["Result"]

# =========================
# Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# Train Random Forest (best model for dataset 2)
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
# SAVE MODEL
# =========================
joblib.dump(rf_model, "rf_dataset2.pkl")

print("✅ Random Forest model for DATASET 2 saved as rf_dataset2.pkl")
