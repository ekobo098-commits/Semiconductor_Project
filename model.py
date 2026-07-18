# ============================================================
# Semiconductor Defect Prediction using XGBoost
# ============================================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from xgboost import XGBClassifier

# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv("5_uci-secom.csv")

print("Dataset Shape:", df.shape)

# ============================================================
# Remove Time Column
# ============================================================

if "Time" in df.columns:
    df.drop(columns=["Time"], inplace=True)

# ============================================================
# Separate Features and Target
# ============================================================

X = df.drop("Pass/Fail", axis=1)

y = df["Pass/Fail"]

# Convert labels
# Pass = 0
# Fail = 1

y = y.replace({
    -1:0,
     1:1
})

# ============================================================
# Remove Constant Columns
# ============================================================

constant_cols = [
    col for col in X.columns
    if X[col].nunique(dropna=False) <= 1
]

print("Constant Columns Removed:", len(constant_cols))

X.drop(columns=constant_cols, inplace=True)

# ============================================================
# Remove Columns with >70% Missing Values
# ============================================================

missing = X.isnull().mean()

remove_cols = missing[missing > 0.70].index

print("Columns Removed (>70% Missing):", len(remove_cols))

X.drop(columns=remove_cols, inplace=True)

# ============================================================
# Median Imputation
# ============================================================

imputer = SimpleImputer(strategy="median")

X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)

# ============================================================
# Train/Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ============================================================
# Handle Class Imbalance
# ============================================================

negative = (y_train == 0).sum()

positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print("Scale Pos Weight:", scale_pos_weight)

# ============================================================
# Build XGBoost Model
# ============================================================

model = XGBClassifier(

    objective="binary:logistic",

    n_estimators=300,

    learning_rate=0.05,

    max_depth=6,

    subsample=0.8,

    colsample_bytree=0.8,

    scale_pos_weight=scale_pos_weight,

    random_state=42,

    eval_metric="logloss"
)

# ============================================================
# Train Model
# ============================================================

model.fit(X_train, y_train)

print("Training Complete.")

# ============================================================
# Predictions
# ============================================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:,1]

# ============================================================
# Evaluation
# ============================================================

print("\n==============================")
print("Model Performance")
print("==============================")

print("Accuracy :", accuracy_score(y_test, y_pred))

print("Precision:", precision_score(y_test, y_pred))

print("Recall   :", recall_score(y_test, y_pred))

print("F1 Score :", f1_score(y_test, y_pred))

print("ROC AUC  :", roc_auc_score(y_test, y_prob))

print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Pass","Fail"]
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.show()

# ============================================================
# Save Model
# ============================================================

joblib.dump(model, "xgboost_model.pkl")

joblib.dump(imputer, "imputer.pkl")

joblib.dump(list(X.columns), "feature_columns.pkl")

print("\nModel Saved Successfully.")

# ============================================================
# Load Model Again
# ============================================================

loaded_model = joblib.load("xgboost_model.pkl")

loaded_imputer = joblib.load("imputer.pkl")

feature_columns = joblib.load("feature_columns.pkl")

print("Model Reloaded Successfully.")

# ============================================================
# Test Saved Model
# ============================================================

sample = X_test.iloc[[0]]

prediction = loaded_model.predict(sample)

probability = loaded_model.predict_proba(sample)

print("\nPrediction:", prediction[0])

print("Probability:")

print(probability)