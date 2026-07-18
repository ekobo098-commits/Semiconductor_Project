# Semiconductor_Project

# ⚡ AI-Based Semiconductor Defect Prediction System

An end-to-end **Industrial Artificial Intelligence** project that predicts semiconductor manufacturing defects using **Extreme Gradient Boosting (XGBoost)** and deploys the trained model through an interactive **Streamlit** dashboard.

This project demonstrates the complete machine learning lifecycle, including data preprocessing, feature engineering, model training, evaluation, explainable insights, and interactive prediction for semiconductor manufacturing quality control.

---

# Project Overview

Semiconductor fabrication facilities generate hundreds of sensor measurements for every manufactured wafer. Detecting defective wafers early can reduce production costs, improve yield, and enhance manufacturing efficiency.

This project develops a machine learning pipeline capable of identifying defective wafers using high-dimensional process sensor data.

The application includes:

- Industrial data preprocessing
- Missing value handling
- Feature selection
- Class imbalance handling
- XGBoost model training
- Model evaluation
- Feature importance visualization
- Interactive wafer defect prediction
- Industrial AI dashboard

---

# Features

- Upload semiconductor manufacturing datasets
- Automatic preprocessing pipeline
- Constant feature removal
- Missing feature elimination
- Median value imputation
- XGBoost classifier training
- Hyperparameter adjustment
- Model evaluation dashboard
- Confusion matrix
- Feature importance visualization
- Interactive wafer prediction
- Real-time defect probability estimation

---

# Dataset

**Dataset**

UCI Machine Learning Repository

**Dataset Name**

SECOM Manufacturing Dataset

**Problem Type**

Binary Classification

Target Variable

| Label | Meaning |
|--------|----------|
| Pass (0) | Non-defective wafer |
| Fail (1) | Defective wafer |

The dataset contains hundreds of semiconductor process sensor measurements collected during manufacturing.

---

# Machine Learning Pipeline

```
Raw Semiconductor Data
            │
            ▼
Remove Time Column
            │
            ▼
Remove Constant Features
            │
            ▼
Remove High Missing Features
            │
            ▼
Median Imputation
            │
            ▼
Train/Test Split
            │
            ▼
Class Imbalance Handling
            │
            ▼
XGBoost Training
            │
            ▼
Performance Evaluation
            │
            ▼
Interactive Prediction Dashboard
```

---

# Technologies Used

- Python
- Streamlit
- XGBoost
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Joblib

---

# Model

Algorithm

**Extreme Gradient Boosting (XGBoost)**

Hyperparameters

- 300 Estimators
- Learning Rate = 0.05
- Maximum Tree Depth = 6
- Subsample = 0.8
- Column Sample = 0.8
- Scale Positive Weight for Class Imbalance

---

# Data Preprocessing

The preprocessing pipeline performs:

- Removal of timestamp column
- Removal of constant features
- Removal of highly incomplete features
- Median value imputation
- Binary label conversion
- Stratified train/test split

---

# Model Evaluation

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Classification Report
- Confusion Matrix

Example Evaluation

| Metric | Score |
|---------|--------|
| Accuracy | 92.68% |
| ROC-AUC | 0.73 |
| Precision | 0.00 |
| Recall | 0.00 |
| F1 Score | 0.00 |

> **Note:** The SECOM dataset is highly imbalanced, with defective wafers representing only a small percentage of samples. While the model achieved high overall accuracy, minority-class detection remains challenging. Future work includes threshold optimization, SMOTE-based resampling, and SHAP explainability.

---

# Dashboard Features

The Streamlit application provides:

- Upload production datasets
- Configure XGBoost hyperparameters
- Train model interactively
- View preprocessing statistics
- Monitor evaluation metrics
- Analyze feature importance
- Perform real-time wafer prediction
- Estimate defect probability

---

# Example Workflow

```
Upload Dataset

↓

Automatic Data Cleaning

↓

Train XGBoost Model

↓

Evaluate Performance

↓

Visualize Feature Importance

↓

Predict Wafer Quality

↓

Estimate Defect Probability
```



# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Semiconductor-Defect-Prediction.git

cd Semiconductor-Defect-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# Future Improvements

- SHAP Explainability
- Threshold Optimization
- SMOTE Oversampling
- Bayesian Hyperparameter Optimization
- Model Versioning
- REST API Deployment
- Real-Time Streaming Predictions
- Manufacturing Execution System (MES) Integration

---

# Industrial AI Relevance

This project demonstrates practical application of machine learning techniques for semiconductor manufacturing quality analysis.

It showcases:

- Industrial sensor analytics
- Predictive quality control
- High-dimensional feature engineering
- Machine learning deployment
- Decision support dashboards
- AI-assisted manufacturing workflows

---

# Disclaimer

This project is intended for research, educational, and portfolio purposes. It uses the publicly available UCI SECOM dataset to demonstrate machine learning techniques applicable to semiconductor manufacturing. Results should not be interpreted as production-ready performance without further validation on operational manufacturing data.

---

# Author

**Sarthak Salve**

Industrial Artificial Intelligence • Machine Learning • Smart Manufacturing • Predictive Analytics
