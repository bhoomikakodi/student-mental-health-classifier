"""
Student Depression Classification - Streamlit Web Application

This interactive Streamlit application demonstrates 5 pre-trained machine learning classifiers
for predicting student depression risk. It supports test dataset upload (CSV), dynamic model
selection, automated calculation of 6 evaluation metrics (Accuracy, ROC-AUC, Precision, Recall,
F1, and MCC), confusion matrix heatmaps, ROC curves, and single-instance risk assessment.
"""

import os
import hashlib
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Depression Risk Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern, Professional Dashboard UI
st.markdown("""
<style>
    /* Global fluid layout */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: clamp(1rem, 3vw, 3rem) !important;
        padding-right: clamp(1rem, 3vw, 3rem) !important;
        max-width: 1400px;
    }

    /* Header hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: clamp(1.2rem, 3vw, 2rem);
        border-radius: 14px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
    }
    .hero-banner h1 {
        color: #ffffff !important;
        font-size: clamp(1.4rem, 2.8vw, 2.2rem) !important;
        margin-bottom: 0.4rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .hero-banner p {
        color: #e0e7ff !important;
        font-size: clamp(0.9rem, 1.4vw, 1.1rem) !important;
        margin-bottom: 0;
        line-height: 1.4;
    }
    
    /* Metric Cards - Adaptive Glassmorphism for Light & Dark */
    .metric-card {
        border-radius: 12px;
        padding: clamp(0.9rem, 1.8vw, 1.3rem) clamp(0.6rem, 1.2vw, 1rem);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 0.8rem;
        min-height: 125px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-title {
        font-size: clamp(0.75rem, 1.1vw, 0.85rem);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: clamp(1.4rem, 2.4vw, 2.0rem);
        font-weight: 800;
        line-height: 1.15;
    }
    .metric-badge {
        display: inline-block;
        font-size: clamp(0.68rem, 0.95vw, 0.75rem);
        padding: 0.2rem 0.55rem;
        border-radius: 9999px;
        font-weight: 600;
        margin-top: 0.35rem;
    }

    /* Light Theme Styles */
    @media (prefers-color-scheme: light) {
        .metric-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        }
        .metric-card:hover { box-shadow: 0 6px 12px rgba(0, 0, 0, 0.09); }
        .metric-title { color: #64748b; }
        .metric-value { color: #0f172a !important; }
        .badge-green { background-color: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
        .badge-blue { background-color: #dbeafe; color: #1e40af; border: 1px solid #bfdbfe; }
        .badge-purple { background-color: #f3e8ff; color: #6b21a8; border: 1px solid #e9d5ff; }
    }

    /* Dark Theme Styles */
    @media (prefers-color-scheme: dark) {
        .metric-card {
            background-color: #1e293b;
            border: 1px solid #334155;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
        }
        .metric-card:hover { box-shadow: 0 6px 14px rgba(0, 0, 0, 0.35); }
        .metric-title { color: #94a3b8; }
        .metric-value { color: #f8fafc !important; }
        .badge-green { background-color: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }
        .badge-blue { background-color: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
        .badge-purple { background-color: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }
    }
    
    /* Responsive media queries */
    @media (max-width: 768px) {
        .metric-card {
            min-height: 105px;
            padding: 0.75rem 0.5rem;
        }
    }
    
    /* Smooth transitions
       - Fast reruns (< ~0.6s, e.g. switching between already-computed models):
         no loader; the new content gets an enter animation (see end of file).
       - Slow reruns (first SVM run) and CSV uploads (while the file is being
         transferred AND while the script reruns on it): the stale content fades
         out and an animated wave loader fades in; the new content fades back in.
       Layout is never changed and the sidebar stays usable. */
    [data-testid="stMain"] [data-testid="stTabs"] {
        opacity: 1;
        transition: opacity 0.5s ease-in-out;
    }
    [data-testid="stApp"]:is([data-test-script-state="running"], :has([data-testid="stFileChipIconSpinner"])) [data-testid="stMain"] [data-testid="stTabs"] {
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.45s ease-in-out 0.6s;
    }
    [data-testid="stMainBlockContainer"] {
        position: relative;
    }
    [data-testid="stApp"]:is([data-test-script-state="running"], :has([data-testid="stFileChipIconSpinner"])) [data-testid="stMain"] [data-testid="stMainBlockContainer"]::before,
    [data-testid="stApp"]:is([data-test-script-state="running"], :has([data-testid="stFileChipIconSpinner"])) [data-testid="stMain"] [data-testid="stMainBlockContainer"]::after {
        content: "";
        position: absolute;
        top: 13rem;
        left: clamp(1rem, 3vw, 3rem);   /* align with the hero banner edges */
        right: clamp(1rem, 3vw, 3rem);
        height: 190px;
        border-radius: 18px;
        overflow: hidden;
        pointer-events: none;
        opacity: 0;
        background-repeat: repeat-x;
        background-size: 800px 100%;
    }
    /* Back wave (light blue) */
    [data-testid="stApp"]:is([data-test-script-state="running"], :has([data-testid="stFileChipIconSpinner"])) [data-testid="stMain"] [data-testid="stMainBlockContainer"]::before {
        height: 170px;
        background-image: url("data:image/svg+xml,%3Csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20width=%27800%27%20height=%27200%27%20viewBox=%270%200%20800%20200%27%20preserveAspectRatio=%27none%27%3E%3Cpath%20fill=%27%2360a5d4%27%20d=%27M0%2C70%20Q100%2C10%20200%2C70%20T400%2C70%20T600%2C70%20T800%2C70%20V200%20H0%20Z%27/%3E%3C/svg%3E");
        animation: loader-fade-in 0.5s ease 0.6s forwards,
                   wave-slide-right 7s linear infinite,
                   wave-bob 2.6s ease-in-out infinite alternate;
    }
    /* Front wave (purple) with label */
    [data-testid="stApp"]:is([data-test-script-state="running"], :has([data-testid="stFileChipIconSpinner"])) [data-testid="stMain"] [data-testid="stMainBlockContainer"]::after {
        content: "Crunching the numbers...";
        box-sizing: border-box;
        padding-top: 128px;
        text-align: center;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        color: #ffffff;
        background-image: url("data:image/svg+xml,%3Csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20width=%27800%27%20height=%27200%27%20viewBox=%270%200%20800%20200%27%20preserveAspectRatio=%27none%27%3E%3Cpath%20fill=%27%236d4fc9%27%20d=%27M0%2C95%20Q66.67%2C50%20133.33%2C95%20T266.67%2C95%20T400%2C95%20T533.33%2C95%20T666.67%2C95%20T800%2C95%20V200%20H0%20Z%27/%3E%3C/svg%3E");
        animation: loader-fade-in 0.5s ease 0.6s forwards,
                   wave-slide-left 4.5s linear infinite,
                   wave-bob 3.4s ease-in-out infinite alternate-reverse;
    }
    @keyframes loader-fade-in {
        from { opacity: 0; }
        to   { opacity: 1; }
    }
    @keyframes wave-slide-right {
        from { background-position-x: 0; }
        to   { background-position-x: 800px; }
    }
    @keyframes wave-slide-left {
        from { background-position-x: 0; }
        to   { background-position-x: -800px; }
    }
    @keyframes wave-bob {
        from { transform: translateY(-6px); }
        to   { transform: translateY(6px); }
    }

    /* Footer text */
    .footer-text {
        text-align: center;
        color: #94a3b8;
        font-size: clamp(0.75rem, 1vw, 0.85rem);
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(148, 163, 184, 0.2);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Asset Loading & Cache Management
# ---------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    """Load pre-trained model artifacts and preprocessor from disk."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "model")
    
    preprocessor_path = os.path.join(model_dir, "preprocessor.joblib")
    metadata_path = os.path.join(model_dir, "metadata.json")
    
    if not os.path.exists(preprocessor_path):
        st.error(f"Preprocessor artifact not found at `{preprocessor_path}`. Please run `python src/train.py` first.")
        st.stop()
        
    preprocessor = joblib.load(preprocessor_path)
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
        
    models = {
        "Logistic Regression": joblib.load(os.path.join(model_dir, "logistic_regression.joblib")),
        "Decision Tree": joblib.load(os.path.join(model_dir, "decision_tree.joblib")),
        "KNN": joblib.load(os.path.join(model_dir, "knn.joblib")),
        "Naive Bayes": joblib.load(os.path.join(model_dir, "naive_bayes.joblib")),
        "Support Vector Machine": joblib.load(os.path.join(model_dir, "svm.joblib")),
    }
    
    metrics_summary_path = os.path.join(model_dir, "metrics_summary.csv")
    metrics_summary_df = pd.read_csv(metrics_summary_path) if os.path.exists(metrics_summary_path) else None
    
    return preprocessor, metadata, models, metrics_summary_df


@st.cache_data
def load_default_test_data():
    """Load default pre-split test dataset from Datasets/."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "Datasets", "test_data.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def get_predictions(model_name, _model, X_proc):
    """Cached predicted labels and probability scores per (model, dataset).

    Keeps reruns fast (e.g. SVM takes ~2s) so stale widgets from the previous
    selection don't linger on screen while switching models.
    """
    preds = _model.predict(X_proc)
    if hasattr(_model, "predict_proba"):
        probs = _model.predict_proba(X_proc)[:, 1]
    elif hasattr(_model, "decision_function"):
        probs = 1 / (1 + np.exp(-_model.decision_function(X_proc)))
    else:
        probs = preds
    return preds, probs


@st.cache_data(show_spinner=False)
def read_dataset_bytes(file_name):
    """Raw bytes of a CSV in Datasets/ for the sidebar download buttons (None if missing)."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "Datasets", file_name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return None


preprocessor, metadata, models, metrics_summary_df = load_model_artifacts()
default_test_df = load_default_test_data()

# ---------------------------------------------------------
# Sidebar: Controls & Data Source
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/brain.png", width=70)
    st.title("Classification Control")
    st.markdown("---")
    
    # 1. Dataset Upload Option
    st.subheader("1. Test Dataset Source")
    uploaded_file = st.file_uploader(
        "Upload Test Data (CSV)",
        type=["csv"],
        help="Upload a test CSV containing predictor features and 'Depression' target column."
    )
    
    if uploaded_file is not None:
        try:
            active_df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded CSV loaded ({active_df.shape[0]} rows, {active_df.shape[1]} cols)")
            data_source_name = f"Uploaded File ({uploaded_file.name})"
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            active_df = default_test_df
            data_source_name = "Default Test Dataset (test_data.csv)"
    else:
        active_df = default_test_df
        data_source_name = "Default Test Dataset (test_data.csv)"
        st.info(f"Using default test dataset ({active_df.shape[0] if active_df is not None else 0} samples)")

    st.markdown("---")
    
    # 2. Model Selection Dropdown
    st.subheader("2. Select ML Classifier")
    model_options = list(models.keys()) + ["Compare All 5 Models"]
    selected_option = st.selectbox(
        "Choose Model to Evaluate",
        model_options,
        index=0,
        help="Select any individual classifier to inspect detailed metrics, confusion matrix, and ROC curve, or choose Compare All."
    )
    
    st.markdown("---")

    # 3. Dataset Downloads
    st.subheader("3. Download Datasets")
    for _label, _file_name in [
        ("⬇️ Test Data (test_data.csv)", "test_data.csv"),
        ("⬇️ Full Dataset (.csv)", "Student_Depression_Dataset.csv"),
    ]:
        _data = read_dataset_bytes(_file_name)
        if _data is not None:
            st.download_button(
                _label,
                data=_data,
                file_name=_file_name,
                mime="text/csv",
                use_container_width=True,
                help=f"Download {_file_name}",
                key=f"download_{_file_name}",
            )

    st.markdown("---")
    st.caption("Student Depression Risk Assessment System")
    st.caption("© 2026 Bhoomika K G")


# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <h1>🧠 Student Depression Risk Classification Dashboard</h1>
    <p>Comparative Evaluation of 5 Machine Learning Classifiers with Live Streamlit Inference</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Validate Active Dataset
# ---------------------------------------------------------
if active_df is None:
    st.error("No test dataset available. Please ensure `test_data.csv` exists or upload a valid CSV.")
    st.stop()

target_col = metadata.get("target_col", "Depression")

if target_col not in active_df.columns:
    st.error(f"Uploaded dataset must contain the target column: `{target_col}`.")
    st.stop()

# Separate features and target
X_active = active_df.drop(columns=[target_col])
y_active = active_df[target_col]

# Check missing required features
missing_features = [col for col in metadata["feature_names"] if col not in X_active.columns]
if missing_features:
    st.error(f"Dataset is missing required features: {missing_features}")
    st.stop()

# Ensure correct column ordering
X_active = X_active[metadata["feature_names"]]

# Transform test data using saved preprocessor
try:
    X_active_proc = preprocessor.transform(X_active)
except Exception as e:
    st.error(f"Error during feature preprocessing: {e}")
    st.stop()


# ---------------------------------------------------------
# App Layout: Tabs for Workflow Navigation
# ---------------------------------------------------------
tab_eval, tab_compare, tab_data, tab_predict = st.tabs([
    "📊 Model Evaluation",
    "📈 Multi-Model Comparison",
    "📁 Dataset Explorer",
    "🔮 Single-Student Prediction"
])


# =========================================================
# TAB 1: MODEL EVALUATION (Individual Model Deep Dive)
# =========================================================
with tab_eval:
    if selected_option == "Compare All 5 Models":
        st.info("💡 You have selected **'Compare All 5 Models'**. Confusion matrices of all models are shown below. Switch to the **'Multi-Model Comparison'** tab for comparative charts, or select an individual model in the sidebar for detailed metrics and ROC curve.")
        st.markdown("### 🧩 Confusion Matrices: All 5 Models")
        st.caption(f"Evaluated on **{len(active_df):,} test samples** from *{data_source_name}* (Inference-only mode).")

        cm_labels = ["No Depression (0)", "Depression (1)"]
        model_items = list(models.items())
        for row_start in range(0, len(model_items), 2):
            cm_cols = st.columns(2)
            for col, (m_name, mdl) in zip(cm_cols, model_items[row_start:row_start + 2]):
                with col:
                    cm_all = confusion_matrix(y_active, get_predictions(m_name, mdl, X_active_proc)[0])
                    fig_cm_all = px.imshow(
                        cm_all,
                        text_auto=True,
                        x=cm_labels,
                        y=cm_labels,
                        labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                        color_continuous_scale="Blues",
                        title=f"Confusion Matrix ({m_name})"
                    )
                    fig_cm_all.update_layout(
                        height=380,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=40, r=40, t=50, b=40),
                        font=dict(family="sans-serif")
                    )
                    st.plotly_chart(fig_cm_all, use_container_width=True)
    else:
        model_to_display = selected_option
        st.markdown(f"### 📊 Performance Metrics: {model_to_display}")
        st.caption(f"Evaluated on **{len(active_df):,} test samples** from *{data_source_name}* (Inference-only mode).")

        # Predict using selected pre-trained model (cached)
        current_model = models[model_to_display]
        y_pred, y_prob = get_predictions(model_to_display, current_model, X_active_proc)

        # Compute 6 required metrics
        acc = accuracy_score(y_active, y_pred)
        auc_val = roc_auc_score(y_active, y_prob)
        prec = precision_score(y_active, y_pred, zero_division=0)
        rec = recall_score(y_active, y_pred, zero_division=0)
        f1 = f1_score(y_active, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_active, y_pred)

        # Display 6 Metric Cards in 2 Responsive Rows
        row1_c1, row1_c2, row1_c3 = st.columns(3)
    
        with row1_c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Accuracy</div>
                <div class="metric-value">{acc:.2%}</div>
                <span class="metric-badge badge-green">Overall Match</span>
            </div>
            """, unsafe_allow_html=True)
        
        with row1_c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">ROC-AUC Score</div>
                <div class="metric-value">{auc_val:.4f}</div>
                <span class="metric-badge badge-blue">Separability</span>
            </div>
            """, unsafe_allow_html=True)
        
        with row1_c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Precision</div>
                <div class="metric-value">{prec:.2%}</div>
                <span class="metric-badge badge-purple">Positive Quality</span>
            </div>
            """, unsafe_allow_html=True)

        row2_c1, row2_c2, row2_c3 = st.columns(3)
        
        with row2_c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Recall (Sensitivity)</div>
                <div class="metric-value">{rec:.2%}</div>
                <span class="metric-badge badge-green">True Positive Rate</span>
            </div>
            """, unsafe_allow_html=True)
        
        with row2_c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">F1 Score</div>
                <div class="metric-value">{f1:.4f}</div>
                <span class="metric-badge badge-blue">Harmonic Mean</span>
            </div>
            """, unsafe_allow_html=True)
        
        with row2_c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">MCC Score</div>
                <div class="metric-value">{mcc:.4f}</div>
                <span class="metric-badge badge-purple">Correlation</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Visualizations: Confusion Matrix and ROC Curve Side-by-Side
        col_viz1, col_viz2 = st.columns(2)

        with col_viz1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_active, y_pred)
            cm_labels = ["No Depression (0)", "Depression (1)"]
        
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                x=cm_labels,
                y=cm_labels,
                labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                color_continuous_scale="Blues",
                title=f"Confusion Matrix ({model_to_display})"
            )
            fig_cm.update_layout(
                width=480,
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=40, r=40, t=50, b=40),
                font=dict(family="sans-serif")
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_viz2:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_active, y_prob)
        
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{model_to_display} (AUC = {auc_val:.4f})',
                line=dict(color='#2563eb', width=3)
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Chance (AUC = 0.50)',
                line=dict(color='#94a3b8', dash='dash')
            ))
            fig_roc.update_layout(
                title=f"ROC Curve ({model_to_display})",
                xaxis_title="False Positive Rate (1 - Specificity)",
                yaxis_title="True Positive Rate (Sensitivity)",
                width=480,
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=40, r=40, t=50, b=40),
                legend=dict(x=0.45, y=0.1)
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        # Classification Report
        st.subheader("Classification Report")
        report_dict = classification_report(y_active, y_pred, target_names=["No Depression (0)", "Depression (1)"], output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(
            report_df.style.format(precision=4).background_gradient(cmap="Blues", subset=["precision", "recall", "f1-score"]),
            use_container_width=True
        )


# =========================================================
# TAB 2: MULTI-MODEL COMPARISON (All 5 Models)
# =========================================================
with tab_compare:
    st.markdown("### 📈 Comparative Evaluation Across All 5 ML Classifiers")
    
    # Calculate live metrics for all 5 models on the active test set
    comparison_records = []
    roc_curves_data = {}
    
    for m_name, mdl in models.items():
        preds, probs = get_predictions(m_name, mdl, X_active_proc)

        c_acc = accuracy_score(y_active, preds)
        c_auc = roc_auc_score(y_active, probs)
        c_prec = precision_score(y_active, preds, zero_division=0)
        c_rec = recall_score(y_active, preds, zero_division=0)
        c_f1 = f1_score(y_active, preds, zero_division=0)
        c_mcc = matthews_corrcoef(y_active, preds)

        comparison_records.append({
            "ML Model Name": m_name,
            "Accuracy": round(c_acc, 4),
            "AUC": round(c_auc, 4),
            "Precision": round(c_prec, 4),
            "Recall": round(c_rec, 4),
            "F1": round(c_f1, 4),
            "MCC": round(c_mcc, 4)
        })
        
        fpr_c, tpr_c, _ = roc_curve(y_active, probs)
        roc_curves_data[m_name] = (fpr_c, tpr_c, c_auc)

    comp_df = pd.DataFrame(comparison_records)

    # 1. Comparison Table
    st.subheader("1. Standard Evaluation Metrics Table")
    st.dataframe(
        comp_df.style.highlight_max(
            subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
            color="#bbf7d0"
        ).format(precision=4),
        use_container_width=True
    )

    # 2. Comparative Bar Charts
    st.subheader("2. Metric Comparison Chart")
    melted_comp = comp_df.melt(
        id_vars=["ML Model Name"],
        value_vars=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
        var_name="Metric",
        value_name="Score"
    )
    
    fig_bar = px.bar(
        melted_comp,
        x="Metric",
        y="Score",
        color="ML Model Name",
        barmode="group",
        title="Side-by-Side Model Performance Across All 6 Evaluation Metrics",
        color_discrete_sequence=px.colors.qualitative.Prism,
        text_auto=".2f"
    )
    fig_bar.update_layout(
        yaxis=dict(range=[0, 1.05]),
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # 3. Combined ROC Curves
    st.subheader("3. Combined ROC Curves")
    fig_multi_roc = go.Figure()
    
    colors = ["#2563eb", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6"]
    for i, (m_name, (fpr_c, tpr_c, a_val)) in enumerate(roc_curves_data.items()):
        fig_multi_roc.add_trace(go.Scatter(
            x=fpr_c, y=tpr_c,
            mode='lines',
            name=f'{m_name} (AUC = {a_val:.4f})',
            line=dict(color=colors[i % len(colors)], width=2.5)
        ))
        
    fig_multi_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Chance (AUC = 0.50)',
        line=dict(color='#94a3b8', dash='dash')
    ))
    
    fig_multi_roc.update_layout(
        title="Combined ROC Curves for All 5 Classifiers",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=480,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(x=0.55, y=0.15)
    )
    st.plotly_chart(fig_multi_roc, use_container_width=True)

    # 4. Model Observations & Winner
    st.markdown("### 📋 Model Observations & Analysis")
    
    obs_data = [
        {"ML Model Name": "Logistic Regression", "Observation": "Strong linear baseline with highest ROC-AUC (0.9184) and balanced Precision (85.76%) and Recall (87.88%). Highly efficient and well-calibrated."},
        {"ML Model Name": "Support Vector Machine (SVM)", "Observation": "Top-tier accuracy (84.38%) and MCC (0.6766). Effectively constructs a non-linear hyper-plane separating healthy vs at-risk students."},
        {"ML Model Name": "K-Nearest Neighbors (KNN)", "Observation": "High recall (89.84%) and AUC (0.9030), successfully clustering students with similar stress, sleep, and academic profiles."},
        {"ML Model Name": "Decision Tree", "Observation": "Interpretable tree structure capturing hierarchical thresholds (e.g., Academic Pressure > 3.5 & Suicidal Thoughts). Solid F1 of 85.78%."},
        {"ML Model Name": "Naive Bayes", "Observation": "Achieves highest sensitivity/recall (97.92%) with AUC of 0.9118. However, lower precision (72.91%) due to conditional independence assumption across correlated pressure features."}
    ]
    st.table(pd.DataFrame(obs_data))

    st.success("🏆 **Overall Winner: Support Vector Machine (SVM) / Logistic Regression** — Both models achieve the highest generalizable accuracy (~84.4%), ROC-AUC (> 0.915), and MCC (0.677), delivering optimal balance between precision and sensitivity for mental health risk screening.")


# =========================================================
# TAB 3: DATASET EXPLORER
# =========================================================
with tab_data:
    st.markdown("### 📁 Active Test Dataset Overview")
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.write(f"**Total Samples:** {len(active_df):,}")
        st.write(f"**Total Features:** {X_active.shape[1]}")
    with col_d2:
        dist_counts = y_active.value_counts()
        st.write(f"**Target Distribution:** Class 1 (Depression): {dist_counts.get(1, 0):,} ({dist_counts.get(1, 0)/len(y_active):.1%}) | Class 0 (No Depression): {dist_counts.get(0, 0):,} ({dist_counts.get(0, 0)/len(y_active):.1%})")

    st.subheader("Dataset Sample Preview")
    st.dataframe(active_df.head(15), use_container_width=True)

    st.subheader("Statistical Summary of Numerical Features")
    st.dataframe(active_df[metadata["num_cols"]].describe().transpose(), use_container_width=True)


# =========================================================
# TAB 4: SINGLE-STUDENT PREDICTION (What-If Interactive Simulation)
# =========================================================
with tab_predict:
    st.markdown("### 🔮 Interactive Student Depression Risk Assessment")
    st.write("Adjust the behavioral and academic parameters below or select a quick profile preset to predict depression risk in real time.")

    # Preset selection
    preset_choice = st.radio(
        "⚡ Quick Profile Presets:",
        ["🚨 High Risk Student Profile", "✅ Healthy / Low Risk Student Profile", "⚙️ Custom Input"],
        horizontal=True
    )

    # Defaults based on preset
    if preset_choice == "🚨 High Risk Student Profile":
        def_acad = 5.0
        def_study_sat = 1.0
        def_fin = 5.0
        def_hours = 11.0
        def_suicidal = "Yes"
        def_sleep = "Less than 5 hours"
        def_diet = "Unhealthy"
        def_fam = "Yes"
        def_cgpa = 6.0
    elif preset_choice == "✅ Healthy / Low Risk Student Profile":
        def_acad = 1.0
        def_study_sat = 5.0
        def_fin = 1.0
        def_hours = 5.0
        def_suicidal = "No"
        def_sleep = "7-8 hours"
        def_diet = "Healthy"
        def_fam = "No"
        def_cgpa = 8.5
    else:
        def_acad = 3.0
        def_study_sat = 3.0
        def_fin = 3.0
        def_hours = 8.0
        def_suicidal = "No"
        def_sleep = "5-6 hours"
        def_diet = "Moderate"
        def_fam = "No"
        def_cgpa = 7.5

    with st.form("single_prediction_form"):
        col_p1, col_p2, col_p3 = st.columns(3)
        
        with col_p1:
            st.markdown("##### 👤 Demographic & Academic")
            p_gender = st.selectbox("Gender", ["Male", "Female"])
            p_age = st.slider("Age", min_value=16, max_value=60, value=22)
            p_city = st.selectbox("City", ["Bangalore", "Mumbai", "Delhi", "Hyderabad", "Chennai", "Kolkata", "Pune", "Others"])
            p_degree = st.selectbox("Degree", ["B.Tech", "B.Sc", "B.Com", "B.A", "M.Tech", "MBA", "M.Sc", "Others"])
            p_profession = st.selectbox("Profession", ["Student", "Software Engineer", "Doctor", "Teacher", "Business", "Others"])
            p_cgpa = st.slider("CGPA", min_value=0.0, max_value=10.0, value=def_cgpa, step=0.1)

        with col_p2:
            st.markdown("##### 📚 Stress & Workload")
            p_acad_press = st.slider("Academic Pressure (1-5)", min_value=1.0, max_value=5.0, value=def_acad, step=1.0)
            p_work_press = st.slider("Work Pressure (0-5)", min_value=0.0, max_value=5.0, value=0.0, step=1.0)
            p_study_sat = st.slider("Study Satisfaction (1-5)", min_value=1.0, max_value=5.0, value=def_study_sat, step=1.0)
            p_job_sat = st.slider("Job Satisfaction (0-5)", min_value=0.0, max_value=5.0, value=0.0, step=1.0)
            p_work_study_hours = st.slider("Work / Study Hours per Day", min_value=1.0, max_value=16.0, value=def_hours, step=0.5)
            p_fin_stress = st.slider("Financial Stress (1-5)", min_value=1.0, max_value=5.0, value=def_fin, step=1.0)

        with col_p3:
            st.markdown("##### 🌿 Lifestyle & Mental Health History")
            sleep_options = ["5-6 hours", "7-8 hours", "Less than 5 hours", "More than 8 hours", "Others"]
            p_sleep = st.selectbox("Sleep Duration", sleep_options, index=sleep_options.index(def_sleep))
            
            diet_options = ["Healthy", "Moderate", "Unhealthy", "Others"]
            p_diet = st.selectbox("Dietary Habits", diet_options, index=diet_options.index(def_diet))
            
            suicidal_options = ["No", "Yes"]
            p_suicidal = st.selectbox("Have you ever had suicidal thoughts?", suicidal_options, index=suicidal_options.index(def_suicidal))
            
            fam_options = ["No", "Yes"]
            p_family_hist = st.selectbox("Family History of Mental Illness", fam_options, index=fam_options.index(def_fam))
            
            p_model_choice = st.selectbox("Select Model for Inference", list(models.keys()), index=0)

        submit_btn = st.form_submit_button("🔍 Assess Depression Risk", use_container_width=True)

    if submit_btn:
        input_data = {
            "Gender": p_gender,
            "Age": float(p_age),
            "City": p_city,
            "Profession": p_profession,
            "Academic Pressure": float(p_acad_press),
            "Work Pressure": float(p_work_press),
            "CGPA": float(p_cgpa),
            "Study Satisfaction": float(p_study_sat),
            "Job Satisfaction": float(p_job_sat),
            "Sleep Duration": p_sleep,
            "Dietary Habits": p_diet,
            "Degree": p_degree,
            "Have you ever had suicidal thoughts ?": p_suicidal,
            "Work/Study Hours": float(p_work_study_hours),
            "Financial Stress": float(p_fin_stress),
            "Family History of Mental Illness": p_family_hist
        }

        input_df = pd.DataFrame([input_data])
        # Ensure order matches
        input_df = input_df[metadata["feature_names"]]
        
        # Preprocess
        input_proc = preprocessor.transform(input_df)
        
        pred_model = models[p_model_choice]
        single_pred = pred_model.predict(input_proc)[0]
        
        if hasattr(pred_model, "predict_proba"):
            single_prob = pred_model.predict_proba(input_proc)[0][1]
        else:
            single_prob = float(single_pred)

        st.markdown("---")
        st.subheader("Assessment Result")
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            if single_pred == 1:
                st.error("🚨 **High Risk of Depression Detected**")
            else:
                st.success("✅ **Low Risk / Healthy Profile**")
                
            st.metric("Predicted Probability of Depression", f"{single_prob:.1%}")
            
        with res_col2:
            st.write(f"**Evaluated by:** {p_model_choice}")
            st.progress(float(single_prob))
            if single_prob > 0.5:
                st.warning("⚠️ Indicators such as high academic pressure, financial stress, or suicidal thoughts contribute to elevated risk scores. Early counseling and stress-management interventions are recommended.")
            else:
                st.info("ℹ️ Behavioral indicators remain within standard ranges. Maintaining healthy sleep duration, balanced study hours, and physical well-being is encouraged.")


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("""
<div class="footer-text">
    © 2026 Bhoomika K G<br>
    Built with Streamlit & Scikit-Learn
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Smooth enter animation: runs at the end of every script run. The keyframe name
# changes whenever the model/dataset changes, which restarts the CSS animation on
# the tab panels so the new content fades in instead of snapping into place.
# ---------------------------------------------------------
_anim_id = hashlib.md5(f"{selected_option}|{data_source_name}".encode()).hexdigest()[:8]
st.markdown(
    f"""
    <style>
    [data-testid="stTabPanel"] {{ animation: tab-enter-{_anim_id} 0.6s ease-out; }}
    @keyframes tab-enter-{_anim_id} {{
        from {{ opacity: 0.35; transform: translateY(10px); }}
        to   {{ opacity: 1;    transform: none; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)
