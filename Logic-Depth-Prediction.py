# logic_depth_pipeline.py
# 📌 ML pipeline to train & predict **combinational logic depth**

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# ========== STEP 1: LOAD & PREPROCESS DATA ==========
print("📤 Using training dataset: combinational_logic_dataset.csv")
train_file = "combinational_logic_dataset.csv"

if not os.path.exists(train_file):
    raise FileNotFoundError(f"❌ {train_file} not found. Please place it in the project folder.")

df = pd.read_csv(train_file)

# Dataset overview
print("\n📊 Dataset Overview:")
print(df.head())
print(f"\n🔹 Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Features (X) and target (y)
X = df.iloc[:, :-1]  # all columns except last
y = df.iloc[:, -1]   # last column as target

# ========== STEP 2: TRAIN ML MODELS ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(random_state=42)
gb_model = GradientBoostingRegressor(random_state=42)

rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# Predictions
rf_pred = rf_model.predict(X_test)
gb_pred = gb_model.predict(X_test)

# Metrics
rf_mae, rf_mse, rf_r2 = mean_absolute_error(y_test, rf_pred), mean_squared_error(y_test, rf_pred), r2_score(y_test, rf_pred)
gb_mae, gb_mse, gb_r2 = mean_absolute_error(y_test, gb_pred), mean_squared_error(y_test, gb_pred), r2_score(y_test, gb_pred)

# ========== STEP 3: MODEL PERFORMANCE ==========
print("\n🏆 Model Performance Summary 🏆")
print("=" * 50)
print(f"📌 Random Forest: MAE={rf_mae:.4f}, MSE={rf_mse:.4f}, R²={rf_r2:.4f}")
print(f"📌 Gradient Boosting: MAE={gb_mae:.4f}, MSE={gb_mse:.4f}, R²={gb_r2:.4f}")

if gb_mae < rf_mae:
    improvement = ((rf_mae - gb_mae) / rf_mae) * 100
    print(f"\n🔥 Gradient Boosting improved MAE by {improvement:.2f}% compared to Random Forest 🔥")

# ========== STEP 4: FEATURE IMPORTANCE ==========
rf_importances = rf_model.feature_importances_
gb_importances = gb_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 5))
sns.barplot(x=feature_names, y=rf_importances, palette="Blues")
plt.title("📊 Feature Importance (Random Forest)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x=feature_names, y=gb_importances, palette="Greens")
plt.title("📊 Feature Importance (Gradient Boosting)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# ========== STEP 5: SAVE MODELS ==========
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(gb_model, 'gb_model.pkl')
print("\n✅ Models saved as 'rf_model.pkl' and 'gb_model.pkl'")

# ========== STEP 6: PREDICTIONS ON NEW DATASET ==========
print("\n📤 Using prediction dataset: combinational_logic_dataset_prediction.csv")
pred_file = "combinational_logic_dataset_prediction.csv"

if not os.path.exists(pred_file):
    raise FileNotFoundError(f"❌ {pred_file} not found. Please place it in the project folder.")

new_df = pd.read_csv(pred_file)

# Match features with training set
original_features = list(X.columns)
new_df = new_df[[col for col in original_features if col in new_df.columns]]

for col in original_features:
    if col not in new_df.columns:
        new_df[col] = X[col].mean()

new_df = new_df[original_features]

# ========== STEP 7: MAKE PREDICTIONS ==========
loaded_rf_model = joblib.load('rf_model.pkl')
new_predictions = loaded_rf_model.predict(new_df)

print("\n📈 Predictions Summary:")
print("=" * 50)
print(f"🔹 Min: {np.min(new_predictions):.2f}")
print(f"🔹 Max: {np.max(new_predictions):.2f}")
print(f"🔹 Mean: {np.mean(new_predictions):.2f}")
print(f"🔹 Std Dev: {np.std(new_predictions):.2f}")

plt.figure(figsize=(8, 5))
sns.histplot(new_predictions, bins=20, kde=True, color="purple")
plt.xlabel("Predicted Logic Depth")
plt.ylabel("Frequency")
plt.title("📊 Distribution of Predicted Combinational Logic Depth")
plt.tight_layout()
plt.show()

df_predictions = pd.DataFrame({"Predicted Logic Depth": new_predictions})
df_predictions.to_csv("predictions_output.csv", index=False)
print("\n📥 Predictions saved as 'predictions_output.csv'")
