# Circuit Timing Predictor (Predict Propagation Delay)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import os

# ========== CONFIG ==========
DATASETS = [
    "combinational_logic_depth_14nm.csv",
    "combinational_logic_depth_28nm.csv",
    "combinational_logic_depth_45nm.csv",
    "combinational_logic_depth_90nm.csv",
    "combinational_logic_depth_180nm.csv"
]  # ✅ just drop all datasets in the repo folder

TARGET_COLUMN = "Propagation delay (register to flip-flop) (ps)"


# ========== HELPERS ==========
def handle_missing_values(df):
    return df.fillna(df.mean())


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "RandomForest": RandomForestRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[name] = {
            "model": model,
            "mae": mean_absolute_error(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
            "predictions": y_pred,
        }
    return results


def visualize_predictions(y_test, y_pred, model_name, dataset_name):
    plt.figure(figsize=(8, 5))
    sns.kdeplot(y_test, label="Actual", fill=True, alpha=0.4)
    sns.kdeplot(y_pred, label="Predicted", fill=True, alpha=0.4)
    plt.title(f"Actual vs Predicted Delay ({model_name} on {dataset_name})")
    plt.legend()
    plt.tight_layout()
    plt.show()


def save_model_and_predictions(model, y_test, y_pred, model_filename, predictions_filename):
    # Save model
    joblib.dump(model, model_filename)
    # Save predictions
    predictions_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    predictions_df.to_csv(predictions_filename, index=False)


# ========== MAIN WORKFLOW ==========
summary_results = []

for dataset in DATASETS:
    if not os.path.exists(dataset):
        print(f"⚠️ Skipping {dataset} (file not found)")
        continue

    print(f"\n📂 Processing dataset: {dataset}")
    df = pd.read_csv(dataset)

    # Check target column
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f'Target column "{TARGET_COLUMN}" not found in {dataset}. '
            f"Available columns: {df.columns.tolist()}"
        )

    # Handle missing values
    df = handle_missing_values(df)

    # Split features/target
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train + evaluate
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    for name, result in results.items():
        print(f"\n{name} Performance on {dataset}:")
        print(f"MAE={result['mae']:.4f}, MSE={result['mse']:.4f}, R²={result['r2']:.4f}")

        # Save model + predictions with dataset name
        base = dataset.replace(".csv", "")
        model_filename = f"{base}_{name}_model.pkl"
        predictions_filename = f"{base}_{name}_predictions.csv"

        save_model_and_predictions(result["model"], y_test, result["predictions"], model_filename, predictions_filename)
        print(f"✅ Saved {model_filename} and {predictions_filename}")

        # Visualization
        visualize_predictions(y_test, result["predictions"], name, base)

        # Collect for summary
        summary_results.append(
            {
                "Dataset": base,
                "Model": name,
                "MAE": result["mae"],
                "MSE": result["mse"],
                "R²": result["r2"],
            }
        )

# Final summary
print("\n📊 Summary Across All Datasets:")
summary_df = pd.DataFrame(summary_results)
print(summary_df)
summary_df.to_csv("summary_results.csv", index=False)
print("📥 Summary saved as 'summary_results.csv'")
