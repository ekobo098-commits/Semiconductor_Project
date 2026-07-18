import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

# Set page layout to wide for scannable dashboard visualization
st.set_page_config(
    page_title="Intelligent Chip Fabrication Yield Center",
    page_icon="⚡",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# APPLICATION HEADER
# -----------------------------------------------------------------------------
st.title("⚡ Real-Time Intelligent Chip Fabrication Yield Analysis")
st.markdown(
    """
    **CHIPS Act Strategic Directive Execution Center**  
    *Natively boosting onshore semiconductor fabrication yield rates by isolating sensor signal anomalies from real-time high-dimensional wafer data.*
    """
)
st.write("---")

# -----------------------------------------------------------------------------
# SIDEBAR CONTROL ROOM & TRAINING
# -----------------------------------------------------------------------------
st.sidebar.header("🛠️ Process Control Configuration")
uploaded_file = st.sidebar.file_uploader(
    "Upload Production Dataset (UCI SECOM CSV Format)", 
    type=["csv"]
)

# Hyperparameter Adjustments
st.sidebar.subheader("Hyperparameter Controls")
n_estimators = st.sidebar.slider("XGBoost Trees", 50, 500, 300, step=50)
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.2, 0.05, step=0.01)
max_depth = st.sidebar.slider("Max Tree Depth", 3, 10, 6)
missing_threshold = st.sidebar.slider("Missing Value Culling Threshold", 0.5, 0.9, 0.7, step=0.05)

# Session state initialization for pipelines and models
if 'model' not in st.session_state:
    st.session_state.model = None
if 'imputer' not in st.session_state:
    st.session_state.imputer = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}

# -----------------------------------------------------------------------------
# INTERNAL FUNCTIONS FOR DATA PROCESSING
# -----------------------------------------------------------------------------
def run_pipeline(dataframe, missing_pct, trees, lr, depth):
    df_clean = dataframe.copy()
    
    # 1. Clean administrative tracking structures
    if "Time" in df_clean.columns:
        df_clean.drop(columns=["Time"], inplace=True)
        
    if "Pass/Fail" not in df_clean.columns:
        st.error("Target anomaly label array 'Pass/Fail' missing from training source.")
        return None
        
    X = df_clean.drop("Pass/Fail", axis=1)
    y = df_clean["Pass/Fail"].replace({-1: 0, 1: 1}) # Align targets to true binary structure[cite: 5]
    
    # 2. Automated Feature Culling & Selection[cite: 5]
    constant_cols = [col for col in X.columns if X[col].nunique(dropna=False) <= 1]
    X.drop(columns=constant_cols, inplace=True)
    
    missing = X.isnull().mean()
    remove_cols = missing[missing > missing_pct].index
    X.drop(columns=remove_cols, inplace=True)
    
    # 3. Fit Median Imputer
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # 4. Train/Test Verification Segment[cite: 5]
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Calculate scale weight based on genuine imbalance ratios[cite: 5]
    negative = (y_train == 0).sum()
    positive = (y_train == 1).sum()
    scale_pos_weight = negative / max(positive, 1)
    
    # 5. Model Execution
    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=trees,
        learning_rate=lr,
        max_depth=depth,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    
    # Performance Evaluation Generation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5,
        "cm": confusion_matrix(y_test, y_pred),
        "removed_const": len(constant_cols),
        "removed_missing": len(remove_cols),
        "remaining_features": X_imputed.shape[1],
        "report": classification_report(y_test, y_pred, output_dict=True)
    }
    
    return model, imputer, list(X_imputed.columns), metrics

# -----------------------------------------------------------------------------
# STAGE 1: SYSTEM ENGINE DATA & MODEL TRAINING
# -----------------------------------------------------------------------------
if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    
    st.subheader("📊 Ingested Fab Telemetry Overview")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Observed Wafers", raw_df.shape[0])
    col_b.metric("Raw Diagnostic Channels", raw_df.shape[1] - 2 if "Time" in raw_df.columns else raw_df.shape[1] - 1)
    
    if "Pass/Fail" in raw_df.columns:
        fail_rate = (raw_df["Pass/Fail"] == 1).sum() / len(raw_df) * 100
        col_c.metric("Baseline Defect (Fail) Rate", f"{fail_rate:.2f}%")
        
    if st.sidebar.button("🚀 Execute Pipeline Training"):
        with st.spinner("Processing High-Dimensional Signal Matrix..."):
            res = run_pipeline(raw_df, missing_threshold, n_estimators, learning_rate, max_depth)
            if res is not None:
                st.session_state.model, st.session_state.imputer, st.session_state.feature_columns, st.session_state.metrics = res
                st.sidebar.success("Yield engine update complete.")

# -----------------------------------------------------------------------------
# STAGE 2: PROCESS DASHBOARD METRICS & ENGINE INSIGHTS
# -----------------------------------------------------------------------------
if st.session_state.model is not None:
    m = st.session_state.metrics
    
    st.header("📈 Model Evaluation & Signal Profiling Dashboard")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Feature Count (Filtered)", m["remaining_features"])
    col2.metric("Accuracy Score", f"{m['Accuracy']:.3f}")
    col3.metric("Precision (Defects)", f"{m['Precision']:.3f}")
    col4.metric("Recall (Defects)", f"{m['Recall']:.3f}")
    col5.metric("ROC AUC Matrix", f"{m['ROC AUC']:.3f}")
    
    st.write("---")
    
    col_chart_1, col_chart_2 = st.columns(2)
    
    with col_chart_1:
        st.subheader("📋 Structural Chaos Analysis (Feature Culling Summary)")
        summary_df = pd.DataFrame({
            "Metric Status Category": ["Constant Signals Purged", "Dead/Missing Channels Removed", "Active Production Features Enforced"],
            "Count": [m["removed_const"], m["removed_missing"], m["remaining_features"]]
        })
        fig, ax = plt.subplots(figsize=(6, 4.3))
        sns.barplot(data=summary_df, x="Count", y="Metric Status Category", palette="viridis", ax=ax)
        ax.set_xlabel("Quantity")
        ax.set_ylabel("")
        st.pyplot(fig)
        
    with col_chart_2:
        st.subheader("🔥 Fab Feature Importance Ranking Top 10")
        importances = st.session_state.model.feature_importances_
        feat_imp = pd.Series(importances, index=st.session_state.feature_columns).sort_values(ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="magma", ax=ax)
        ax.set_xlabel("Relative Importance Weight")
        st.pyplot(fig)

    st.write("---")

    # -----------------------------------------------------------------------------
    # STAGE 3: REAL-TIME IN-LINE WAFER INSPECTION & ROOT CAUSE MAPPING
    # -----------------------------------------------------------------------------
    st.header("🔬 Inline Scanner Simulation (Real-Time Yield Estimation)")
    st.markdown("Manually isolate single tool instances to confirm process control boundaries or diagnose specific out-of-control conditions.")
    
    # Generate manual entry widgets dynamically for the top 4 highly volatile drivers
    top_drivers = list(feat_imp.index[:4])
    st.write(f"Adjust critical fab metric control limits for driving sensors:")
    
    input_values = {}
    col_input_space = st.columns(4)
    
    for idx, feature in enumerate(st.session_state.feature_columns):
        if feature in top_drivers:
            grid_pos = top_drivers.index(feature)
            with col_input_space[grid_pos]:
                input_values[feature] = st.number_input(f"Sensor Matrix Channel {feature}", value=0.0, step=0.1)
        else:
            input_values[feature] = 0.0

    if st.button("🔍 Run Diagnostics on Current Substrate"):
        # Format prediction sample array matching exact expected model layout
        sample_df = pd.DataFrame([input_values])[st.session_state.feature_columns]
        
        # Inference Generation
        pred = st.session_state.model.predict(sample_df)[0]
        prob = st.session_state.model.predict_proba(sample_df)[0][1]
        
        st.write("### Production Scanner Verdict")
        if pred == 1:
            st.error(f"🚨 **ANOMALOUS WAFER DETECTED**: System calculated target threat probability at **{prob*100:.2f}%**.")
            st.markdown(
                f"""> **Root Cause Localization Analysis:**  
                Process metrics deviate aggressively from baseline targets. Primary structural degradation driven via channel: **Sensor {top_drivers[0]}**."""
            )
        else:
            st.success(f"✅ **WAFER VERIFIED INTEGRAL**: Current product falls cleanly within tolerance zones. Defect probability: **{prob*100:.2f}%**.")
            st.info("💡 Process adjustments fall safely inside planned standard target deviation parameters.")

else:
    # Initial dynamic user interface view before file provisioning
    st.info("💡 System Ready. Upload a telemetry profile (CSV matching UCI SECOM structural configuration) within the Process Control panel to generate model layers.")