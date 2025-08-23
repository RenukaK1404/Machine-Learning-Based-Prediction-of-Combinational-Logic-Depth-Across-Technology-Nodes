# circuit_reliability_pipeline.py
# 📌 Circuit Reliability Checker (Slack Violation Classifier)
# Predicts Slack Violation Classification (0 = safe, 1 = violation)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.utils import resample
import joblib
import os

# ========== CONFIG ==========
DATASETS = [
    "combinational_logic_depth_14nm.csv",
    "combinational_logic_depth_28nm.csv",
    
]  # ✅ drop datasets here

# ========== HELPERS ==========
def calculate_negative_slack(df):
    if "Propagation delay (register to flip-flop) (ps)" not in df.columns or "Setup time (ps)" not in df.columns:
        raise KeyError("Required columns ('Propagation delay', 'Setup time') not found in dataset.")
    df["Negative Slack (ps)"] = df["Propagation delay (register to flip-flop) (ps)"] - df["Setup time (ps)"]
    df["Negative Slack (ps)"] = np.where(df["Negative Slack (ps)"] < 0, 1, 0)  # 1 = violation, 0 = safe
    return df


def balance_classes(df):
    counts = df["Negative Slack (ps)"].value_counts()
    if counts.min() / counts.max() < 0.5:  # imbalance check
        majority = df[df["Negative Slack (ps)"] == counts.idxmax()]
        minority = df[df["Negative Slack (ps)"] == counts.idxmin()]
        minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
        df = pd.concat([majority, minority_upsampled])
    return df


def evaluate_model(y_test, y_pred, model_name, dataset_name):
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred)
    }
    print(f"\n📊 {model_name} on {dataset_name}: {metrics}")
    return metrics


def plot_confusion(y_test, y_pred, model_name, dataset_name):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix: {model_name} on {dataset_name}")
    plt.show()


def save_outputs(model, y_test, y_pred, base, model_name):
    model_filename = f"{base}_{model_name}_model.pkl"
    preds_filename = f"{base}_{model_name}_predictions.csv"

    joblib.dump(model, model_filename)
    pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).to_csv(preds_filename, index=False)
    print(f"✅ Saved {model_filename}, {preds_filename}")


# ========== MAIN PIPELINE ==========
summary = []

for dataset in DATASETS:
    if not os.path.exists(dataset):
        print(f"⚠️ Skipping {dataset} (file not found)")
        continue

    print(f"\n📂 Processing {dataset}")
    df = pd.read_csv(dataset)

    # Step 1: Add target column (Slack violation)
    df = calculate_negative_slack(df)

    # Step 2: Handle imbalance
    df = balance_classes(df)

    # Step 3: Split features/target
    X = df.drop(columns=["Negative Slack (ps)"])
    y = df["Negative Slack (ps)"]

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Step 4: Train models
    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate
        metrics = evaluate_model(y_test, y_pred, name, dataset.replace(".csv", ""))
        metrics.update({"Dataset": dataset.replace(".csv", ""), "Model": name})
        summary.append(metrics)

        # Save
        save_outputs(model, y_test, y_pred, dataset.replace(".csv", ""), name)

        # Confusion matrix plot
        plot_confusion(y_test, y_pred, name, dataset.replace(".csv", ""))

# Save summary
summary_df = pd.DataFrame(summary)
print("\n📊 Final Summary Across Datasets:")
print(summary_df)
summary_df.to_csv("reliability_summary.csv", index=False)
print("📥 Saved 'reliability_summary.csv'")
