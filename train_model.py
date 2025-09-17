# train_model.py
"""
Complete Diabetes Prediction Project
- Loads dataset
- Cleans & preprocesses
- Trains RandomForest model
- Shows metrics & plots
- Saves trained model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)

# 1) Load dataset
print("Loading dataset...")
df = pd.read_csv("data/diabetes.csv")
print("Dataset shape:", df.shape)
print(df.head())

# 2) Data Cleaning
cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

print("\nMissing values per column:")
print(df.isna().sum())

# 3) Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 4) Train-Test Split
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5) Build Pipeline
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(random_state=42, class_weight="balanced"))
])

# 6) Train Model
print("\nTraining model...")
pipeline.fit(X_train, y_train)

# 7) Evaluate Model
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:,1]

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Diabetes","Diabetes"],
            yticklabels=["No Diabetes","Diabetes"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0,1],[0,1], linestyle="--", color="grey")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Feature Importance
clf = pipeline.named_steps["clf"]
importances = clf.feature_importances_
feature_names = X.columns

fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(x=fi.values, y=fi.index, palette="viridis")
plt.title("Feature Importance (RandomForest)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

# 8) Save Model
joblib.dump(pipeline, "diabetes_model.pkl")
print("\n✅ Model saved as diabetes_model.pkl")
