import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# =====================
# Load dataset 3
# =====================
df = pd.read_csv("data/web-page-phishing.csv")

# Label column
LABEL = "phishing"

X = df.drop(columns=[LABEL])
y = df[LABEL]

# SAVE FEATURE NAMES (CRITICAL)
FEATURE_COLUMNS = list(X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# SAVE AS BUNDLE (MODEL + FEATURES)
bundle = {
    "model": model,
    "features": FEATURE_COLUMNS
}

joblib.dump(bundle, "models/dataset3_features.pkl")

print("✅ dataset3_features.pkl created correctly")
print("Features:", FEATURE_COLUMNS)
