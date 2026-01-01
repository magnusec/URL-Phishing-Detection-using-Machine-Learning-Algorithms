import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# =========================
# Load dataset
# =========================
df = pd.read_csv("data/Phishing_Legitimate_full.csv")

# Drop ID column to prevent data leakage
if "id" in df.columns:
    df = df.drop(columns=["id"])

# Separate features and label
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
# Models
# =========================
models = {
    "Decision Tree": DecisionTreeClassifier(
        criterion="entropy",
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),
    "SVM": SVC(kernel="rbf"),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB()
}

# =========================
# Evaluation
# =========================
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    })

# =========================
# Results table
# =========================
results_df = pd.DataFrame(results)

print("\nFINAL MODEL METRICS COMPARISON:\n")
print(results_df)

# Save results
results_df.to_csv("model_full_metrics_comparison.csv", index=False)
print("\nSaved as model_full_metrics_comparison.csv")
