# ==============================================================================
# STUDENT PERFORMANCE PREDICTION SYSTEM (ML PIPELINE)
# ==============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

# Set visualization style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

# ==============================================================================
# PHASE 1: PROBLEM UNDERSTANDING
# ==============================================================================
"""
Business Problem: Predict student math scores based on demographic attributes, 
parental education, lunch type, and test preparation.
Target Variable: 'math score' (Continuous Numerical Variable -> Regression Task)
Features: Categorical (gender, race, parent education, lunch, prep course) 
          + Numerical (reading score, writing score)
"""
print("=" * 60)
print("PHASE 1: PROBLEM UNDERSTANDING COMPLETED")
print("=" * 60)


# ==============================================================================
# PHASE 2: DATA COLLECTION & INSPECTION
# ==============================================================================
df = pd.read_csv("StudentsPerformance.csv")

print("\n--- Dataset Overview ---")
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 5 Rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())


# ==============================================================================
# PHASE 3: DATA PREPROCESSING
# ==============================================================================
print("\n" + "=" * 60)
print("PHASE 3: DATA PREPROCESSING")
print("=" * 60)

# 1. Handle Missing Values
print("\nMissing Values Count:\n", df.isnull().sum())
# If missing values existed: df.fillna(...) or df.dropna(...)

# 2. Remove Duplicates
duplicate_count = df.duplicated().sum()
print(f"\nDuplicates found: {duplicate_count}")
if duplicate_count > 0:
    df = df.drop_duplicates()
    print("Duplicates removed.")

# 3. Create Feature Copy for Preprocessing
df_processed = df.copy()

# 4. Outlier Treatment (IQR Method on Numerical Features)
num_cols = ["math score", "reading score", "writing score"]
for col in num_cols:
    Q1 = df_processed[col].quantile(0.25)
    Q3 = df_processed[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Capping outliers
    df_processed[col] = np.where(
        df_processed[col] < lower_bound,
        lower_bound,
        np.where(df_processed[col] > upper_bound, upper_bound, df_processed[col]),
    )
print("\nOutliers capped using IQR method.")


# ==============================================================================
# PHASE 4: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
print("\n" + "=" * 60)
print("PHASE 4: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)

# 1. Univariate Analysis - Histograms
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(df["math score"], kde=True, ax=axes[0], color="skyblue").set_title(
    "Math Score Distribution"
)
sns.histplot(
    df["reading score"], kde=True, ax=axes[1], color="salmon"
).set_title("Reading Score Distribution")
sns.histplot(
    df["writing score"], kde=True, ax=axes[2], color="lightgreen"
).set_title("Writing Score Distribution")
plt.tight_layout()
plt.savefig("univariate_analysis.png")
plt.show()

# 2. Bivariate Analysis - Boxplots & Scatter Plots
plt.figure(figsize=(8, 5))
sns.boxplot(x="parental level of education", y="math score", data=df)
plt.xticks(rotation=45)
plt.title("Math Score vs Parental Education")
plt.tight_layout()
plt.savefig("bivariate_boxplot.png")
plt.show()

# 3. Correlation Heatmap (Numerical Columns)
plt.figure(figsize=(6, 4))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()


# ==============================================================================
# PHASE 5: FEATURE ENGINEERING & DATA PREPARATION
# ==============================================================================
print("\n" + "=" * 60)
print("PHASE 5: FEATURE ENGINEERING")
print("=" * 60)

# 1. Feature Creation: Average Score & Total Score
df_processed["total_score"] = (
    df_processed["math score"]
    + df_processed["reading score"]
    + df_processed["writing score"]
)
df_processed["average_score"] = df_processed["total_score"] / 3

print("\nNew Features Added: total_score, average_score")

# 2. Categorical Encoding (One-Hot Encoding)
cat_cols = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
]
df_encoded = pd.get_dummies(df_processed, columns=cat_cols, drop_first=True)

# 3. Separate Features (X) and Target (y)
# Let's predict 'math score' using other features (excluding derived scores to avoid data leakage)
X = df_encoded.drop(
    columns=["math score", "total_score", "average_score"], errors="ignore"
)
y = df_encoded["math score"]

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Feature Scaling / Normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")


# ==============================================================================
# PHASE 6: MODEL BUILDING
# ==============================================================================
print("\n" + "=" * 60)
print("PHASE 6: MODEL BUILDING")
print("=" * 60)

# Training 3 Models as per requirements
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}

trained_models = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    print(f"[✓] {name} trained successfully.")


# ==============================================================================
# PHASE 7: MODEL EVALUATION
# ==============================================================================
print("\n" + "=" * 60)
print("PHASE 7: MODEL EVALUATION")
print("=" * 60)

evaluation_results = []

for name, model in trained_models.items():
    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    evaluation_results.append(
        {"Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r2}
    )

# Display Evaluation Table
results_df = pd.DataFrame(evaluation_results)
print("\n", results_df.to_string(index=False))

# Highlight Best Model based on R2 Score
best_model_name = results_df.sort_values(by="R2 Score", ascending=False).iloc[0][
    "Model"
]
print(f"\n🌟 Best Performing Model: {best_model_name}")
# ==============================================================================
# PHASE 8: MODEL SAVING (USING STANDARD PICKLE)
# ==============================================================================
import pickle

print("\n" + "=" * 60)
print("PHASE 8: SAVING ARTIFACTS WITH PICKLE")
print("=" * 60)

# Best Model Object Ko Extract Karein
best_model_obj = trained_models[best_model_name]

# 1. Save Best Model
with open("best_student_model.pkl", "wb") as f:
    pickle.dump(best_model_obj, f)

# 2. Save Scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# 3. Save Feature Columns List
with open("feature_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print(f"[✓] Best Model ({best_model_name}) saved to 'best_student_model.pkl'")
print("[✓] Scaler saved to 'scaler.pkl'")
print("[✓] Feature Columns saved to 'feature_columns.pkl'")
import pickle

# Linear Regression ya Random Forest model ko direct save karein
best_model_obj = trained_models["Linear Regression"]  # ya "Random Forest"

with open("best_student_model.pkl", "wb") as f:
  pickle.dump(best_model_obj, f)

with open("scaler.pkl", "wb") as f:
  pickle.dump(scaler, f)

with open("feature_columns.pkl", "wb") as f:
  pickle.dump(X.columns.tolist(), f)

print("Nayi pickle files generate ho gayi hain!")