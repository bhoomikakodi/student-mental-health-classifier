"""
Model Training and Evaluation Pipeline for Student Depression Classification.

This script loads the Student Depression dataset, performs data preprocessing and feature
engineering, trains 5 classification models (Logistic Regression, Decision Tree, KNN,
Naive Bayes, SVM), evaluates each model across 6 standardized performance metrics,
and serializes all models and preprocessing artifacts for the Streamlit web application.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load and perform initial validation on the dataset."""
    logging.info(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Remove surrogate ID column if present
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    
    logging.info(f"Dataset loaded successfully with shape: {df.shape}")
    return df


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list, list]:
    """
    Construct ColumnTransformer for numerical imputation + scaling
    and categorical imputation + one-hot encoding.
    """
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    logging.info(f"Numerical columns ({len(num_cols)}): {num_cols}")
    logging.info(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ]
    )

    return preprocessor, num_cols, cat_cols


def train_and_evaluate():
    """Main execution function to train 5 classifiers and export artifacts."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "Datasets", "Student_Depression_Dataset.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join(base_dir, "Datasets", "Student Depression Dataset.csv")
    model_dir = os.path.join(base_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    # 1. Load data
    df = load_dataset(data_path)
    target_col = "Depression"
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Stratified Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Export test_data.csv to Datasets/
    test_df = pd.concat([X_test, y_test], axis=1)
    datasets_test_path = os.path.join(base_dir, "Datasets", "test_data.csv")
    test_df.to_csv(datasets_test_path, index=False)
    logging.info(f"Saved test dataset to {datasets_test_path} with {len(test_df)} rows.")

    # 3. Fit Preprocessing Pipeline
    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Save Preprocessor Artifact
    preprocessor_path = os.path.join(model_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    logging.info(f"Saved preprocessor to {preprocessor_path}")

    # Metadata for app inference
    metadata = {
        "target_col": target_col,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "feature_names": X.columns.tolist()
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # 4. Initialize 5 Required Classifiers
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=7, min_samples_leaf=10, random_state=42
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=15, weights="distance", n_jobs=-1
        ),
        "Naive Bayes": GaussianNB(
            var_smoothing=1e-1
        ),
        "Support Vector Machine": SVC(
            kernel="rbf", C=1.0, probability=True, random_state=42
        )
    }

    model_filenames = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "KNN": "knn.joblib",
        "Naive Bayes": "naive_bayes.joblib",
        "Support Vector Machine": "svm.joblib"
    }

    metrics_records = []
    detailed_results = {}

    logging.info("Training and evaluating classifiers...")

    for model_name, model in models.items():
        logging.info(f"Training {model_name}...")
        
        # Subsample for computationally intensive non-parametric models (SVM, KNN)
        # to ensure instant inference and lean repository size for Streamlit Cloud
        if model_name == "Support Vector Machine":
            sample_indices = np.random.RandomState(42).choice(
                len(X_train_proc), size=min(6000, len(X_train_proc)), replace=False
            )
            model.fit(X_train_proc[sample_indices], y_train.iloc[sample_indices])
        elif model_name == "KNN":
            sample_indices = np.random.RandomState(42).choice(
                len(X_train_proc), size=min(8000, len(X_train_proc)), replace=False
            )
            model.fit(X_train_proc[sample_indices], y_train.iloc[sample_indices])
        else:
            model.fit(X_train_proc, y_train)

        # Predictions on unseen test data
        y_pred = model.predict(X_test_proc)
        y_prob = model.predict_proba(X_test_proc)[:, 1] if hasattr(model, "predict_proba") else None

        # Compute 6 required metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        # Save model artifact
        save_path = os.path.join(model_dir, model_filenames[model_name])
        joblib.dump(model, save_path)
        logging.info(f"Saved {model_name} model to {save_path}")

        metrics_records.append({
            "ML Model Name": model_name,
            "Accuracy": round(acc, 4),
            "AUC": round(auc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "MCC": round(mcc, 4)
        })

        detailed_results[model_name] = {
            "accuracy": float(acc),
            "auc": float(auc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "mcc": float(mcc),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }

    # 5. Export Metrics Summary
    metrics_df = pd.DataFrame(metrics_records)
    metrics_csv_path = os.path.join(model_dir, "metrics_summary.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)

    with open(os.path.join(model_dir, "detailed_results.json"), "w") as f:
        json.dump(detailed_results, f, indent=2)

    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE COMPARISON ON TEST SET (5,581 samples)")
    print("=" * 80)
    print(metrics_df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    train_and_evaluate()
