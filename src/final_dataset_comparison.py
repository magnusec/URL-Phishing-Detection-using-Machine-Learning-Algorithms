import pandas as pd

final_results = {
    "Dataset": [
        "Dataset 1 (Phishing_Legitimate_full)",
        "Dataset 2 (phishing_dataset)",
        "Dataset 3 (web-page-phishing)"
    ],
    "Dataset Size": [
        "10,000",
        "Medium",
        "100,000+"
    ],
    "Best Model": [
        "Random Forest",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        0.9775,
        0.9749,
        0.8814
    ],
    "Remarks": [
        "Balanced and clean dataset",
        "Different feature schema",
        "Real-world, imbalanced dataset"
    ]
}

df = pd.DataFrame(final_results)

print("\nFINAL DATASET COMPARISON\n")
print(df)

df.to_csv("final_dataset_comparison.csv", index=False)
print("\nSaved as final_dataset_comparison.csv")
