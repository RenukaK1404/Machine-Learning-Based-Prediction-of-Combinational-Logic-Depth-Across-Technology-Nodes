#Logic Depth Prediction
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from google.colab import files
import io

# ========== STEP 1: UPLOAD & PREPROCESS DATA ==========
print("📤 Upload your dataset (CSV file)...")
uploaded = files.upload()

# Read the uploaded file
filename = list(uploaded.keys())[0]
df = pd.read_csv(io.BytesIO(uploaded[filename]))

# Display dataset overview
print("\n📊 Dataset Overview:")
print(df.head())
print(f"\n🔹 Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

# Handle missing values by filling with mean
df.fillna(df.mean(), inplace=True)

# Separate features (X) and target (y)
X = df.iloc[:, :-1]  # All columns except the last
y = df.iloc[:, -1]   # Last column as target

# ========== STEP 2: TRAIN ML MODELS ==========
# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
rf_model = RandomForestRegressor(random_state=42)
gb_model = GradientBoostingRegressor(random_state=42)

rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# Evaluate models
rf_pred = rf_model.predict(X_test)
gb_pred = gb_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

gb_mae = mean_absolute_error(y_test, gb_pred)
gb_mse = mean_squared_error(y_test, gb_pred)
gb_r2 = r2_score(y_test, gb_pred)

# ========== STEP 3: DISPLAY MODEL PERFORMANCE ==========
print("\n🏆 Model Performance Summary 🏆")
print("=" * 50)
print(f"📌 Random Forest:")
print(f"   - MAE: {rf_mae:.4f}")
print(f"   - MSE: {rf_mse:.4f}")
print(f"   - R² Score: {rf_r2:.4f}")
print("-" * 50)
print(f"📌 Gradient Boosting:")
print(f"   - MAE: {gb_mae:.4f}")
print(f"   - MSE: {gb_mse:.4f}")
print(f"   - R² Score: {gb_r2:.4f}")

if gb_mae < rf_mae:
    improvement = ((rf_mae - gb_mae) / rf_mae) * 100
    print(f"\n🔥 Gradient Boosting improved MAE by {improvement:.2f}% compared to Random Forest 🔥")

# ========== STEP 4: FEATURE IMPORTANCE VISUALIZATION ==========
rf_importances = rf_model.feature_importances_
gb_importances = gb_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 5))
sns.barplot(x=feature_names, y=rf_importances, palette="Blues")
plt.title("📊 Feature Importance (Random Forest)")
plt.ylabel("Importance Score")
plt.xlabel("Feature")
plt.xticks(rotation=30)
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x=feature_names, y=gb_importances, palette="Greens")
plt.title("📊 Feature Importance (Gradient Boosting)")
plt.ylabel("Importance Score")
plt.xlabel("Feature")
plt.xticks(rotation=30)
plt.show()

# ========== STEP 5: SAVE TRAINED MODELS ==========
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(gb_model, 'gb_model.pkl')

files.download('rf_model.pkl')
files.download('gb_model.pkl')

# ========== STEP 6: UPLOAD NEW DATASET FOR PREDICTIONS ==========
print("\n📤 Upload a new dataset for predictions:")
uploaded = files.upload()

# Read the new dataset
new_filename = list(uploaded.keys())[0]
new_df = pd.read_csv(io.BytesIO(uploaded[new_filename]))

# Ensure feature set matches original dataset
original_features = list(X.columns)

# ✅ Keep only matching columns
new_df = new_df[[col for col in original_features if col in new_df.columns]]

# ✅ Fill missing columns with training data mean
for col in original_features:
    if col not in new_df.columns:
        new_df[col] = X[col].mean()

# ✅ Ensure correct column order
new_df = new_df[original_features]

# ========== STEP 7: MAKE PREDICTIONS ==========
loaded_rf_model = joblib.load('rf_model.pkl')
new_predictions = loaded_rf_model.predict(new_df)

# Display prediction summary
print("\n📈 Predictions Summary:")
print("=" * 50)
print(f"🔹 Min Predicted Depth: {np.min(new_predictions):.2f}")
print(f"🔹 Max Predicted Depth: {np.max(new_predictions):.2f}")
print(f"🔹 Mean Predicted Depth: {np.mean(new_predictions):.2f}")
print(f"🔹 Standard Deviation: {np.std(new_predictions):.2f}")

# Plot prediction distribution
plt.figure(figsize=(8, 5))
sns.histplot(new_predictions, bins=20, kde=True, color="purple")
plt.xlabel("Predicted Logic Depth")
plt.ylabel("Frequency")
plt.title("📊 Distribution of Predicted Combinational Logic Depth")
plt.show()

# Save predictions to CSV
df_predictions = pd.DataFrame({"Predicted Logic Depth": new_predictions})
df_predictions.to_csv("predictions_output.csv", index=False)
files.download("predictions_output.csv")
print("\n📥 Predictions saved as 'predictions_output.csv'")
