# Student Depression Risk Classification & Comparative ML Study

**BITS Pilani — Work Integrated Learning Programmes Division**  
**M.Tech (AIML) — Machine Learning Assignment 2**  
**Course Code / Semester:** Machine Learning (Assignment - 2)  
**Deployment Platform:** Streamlit Community Cloud  

---

## 1. Problem Statement

Depression and mental health challenges among university students have become an escalating global concern, adversely impacting academic performance, retention, and overall personal well-being. Early identification of students susceptible to depression allows academic counselors and healthcare personnel to intervene proactively with targeted support systems.

The objective of this project is to develop and systematically evaluate **five supervised classification machine learning models** to predict whether a student is at risk of depression based on multifaceted lifestyle, academic, and psychological attributes. The project culminates in an interactive, responsive **Streamlit Web Application** deployed to Streamlit Community Cloud, enabling real-time risk inference and multi-model performance benchmarking without requiring local installation or model retraining.

---

## 2. Dataset Description `[1 Mark]`

- **Dataset Name:** Student Depression Dataset
- **Source:** [Kaggle — Student Depression Dataset](https://www.kaggle.com/) / UCI-style Survey Records
- **Total Instances:** 27,901 rows ($\ge 500$ criteria satisfied)
- **Total Features:** 16 predictor features + 1 target variable ($\ge 12$ criteria satisfied)
- **Target Variable:** `Depression` (Binary: `0` = No Depression, `1` = Depression Present)
- **Class Balance:** 
  - Class `1` (Depression): 58.5% (16,336 samples)
  - Class `0` (No Depression): 41.5% (11,565 samples)

### Feature Breakdown

| Feature Name | Data Type | Nature | Description / Domain Range |
| :--- | :--- | :--- | :--- |
| **Gender** | Categorical | Demographic | Gender identity (`Male`, `Female`) |
| **Age** | Numeric (Continuous) | Demographic | Student age (16 – 60 years) |
| **City** | Categorical | Geographic | City of residence / campus location (52 unique cities) |
| **Profession** | Categorical | Socio-economic | Current occupation / student status (`Student`, `Engineer`, etc.) |
| **Academic Pressure** | Numeric (Discrete) | Academic | Self-reported academic workload pressure (Scale: 1 – 5) |
| **Work Pressure** | Numeric (Discrete) | Professional | Workplace or internship pressure (Scale: 0 – 5) |
| **CGPA** | Numeric (Continuous) | Academic | Cumulative Grade Point Average (Scale: 0.0 – 10.0) |
| **Study Satisfaction** | Numeric (Discrete) | Academic | Student satisfaction with current studies (Scale: 1 – 5) |
| **Job Satisfaction** | Numeric (Discrete) | Professional | Satisfaction with employment/internship (Scale: 0 – 5) |
| **Sleep Duration** | Categorical (Ordinal) | Lifestyle | Average daily sleep (`<5 hours`, `5-6 hours`, `7-8 hours`, `>8 hours`, `Others`) |
| **Dietary Habits** | Categorical | Lifestyle | Nutritional quality (`Healthy`, `Moderate`, `Unhealthy`, `Others`) |
| **Degree** | Categorical | Education | Degree level/program (`B.Tech`, `M.Tech`, `B.Sc`, `MBA`, etc.) |
| **Have you ever had suicidal thoughts ?** | Categorical (Binary) | Psychological | History of suicidal ideation (`Yes`, `No`) |
| **Work/Study Hours** | Numeric (Continuous) | Academic | Combined daily hours dedicated to work and studying (1 – 16 hrs) |
| **Financial Stress** | Numeric (Discrete) | Socio-economic | Level of financial strain (Scale: 1 – 5) |
| **Family History of Mental Illness** | Categorical (Binary) | Genetic / Clinical | Family history of psychiatric / mental health conditions (`Yes`, `No`) |
| **Depression (Target)** | Binary Integer | Clinical Target | Risk label (`1`: Depressed, `0`: Non-depressed) |

---

## 3. GitHub Repository Link `[1 Mark]`

- **GitHub Repository URL:** [https://github.com/bhoomikakodi/student-mental-health-classifier](https://github.com/bhoomikakodi/student-mental-health-classifier)
- **Live Streamlit Cloud URL:** `https://student-depression-ml-assignment2.streamlit.app`

*(Note: Replace placeholders with your final published repository and app URLs before creating your submission PDF).*

---

## 4. Models Used & Performance Comparison Table `[5 Marks]`

All 5 classifiers were evaluated on the identical unseen test set (**5,581 stratified test instances**, 20% holdout split) across all six mandatory evaluation metrics:

| ML Model Name | Accuracy | AUC (ROC) | Precision | Recall | F1 Score | MCC Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8436** | **0.9184** | **0.8576** | **0.8788** | **0.8681** | **0.6764** |
| **Decision Tree** | 0.8316 | 0.8991 | 0.8481 | 0.8678 | 0.8578 | 0.6516 |
| **KNN (K-Nearest Neighbors)** | 0.8316 | 0.9030 | 0.8284 | 0.8984 | 0.8620 | 0.6503 |
| **Naive Bayes (Gaussian NB)** | 0.7748 | 0.9118 | 0.7291 | **0.9792** | 0.8358 | 0.5591 |
| **Support Vector Machine (SVM)** | **0.8438** | **0.9157** | **0.8568** | **0.8804** | **0.8684** | **0.6766** |

---

## 5. Observations on Model Performance `[3 Marks]`

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieves the highest ROC-AUC score (**0.9184**) and near-peak accuracy (**84.36%**). By leveraging standardized continuous features and one-hot encoded categories, the linear decision boundary efficiently separates high-risk vs low-risk profiles without overfitting. |
| **Decision Tree** | Delivers strong interpretability with **83.16% accuracy** and **0.8578 F1 score**. Pruning to a maximum depth of 7 effectively identified primary hierarchical split points (`Academic Pressure > 3.5`, `Suicidal Thoughts == Yes`), preventing deep-tree variance. |
| **KNN** | Yields **83.16% accuracy** and **89.84% recall**. Distance-weighted voting across standardized feature space successfully clusters students with analogous academic stress and sleep deprivation patterns. |
| **Naive Bayes** | Demonstrates the highest recall (**97.92%**) and a solid ROC-AUC (**0.9118**), effectively acting as a high-sensitivity screening filter. However, precision (**72.91%**) is slightly degraded due to its feature independence assumption across correlated metrics (e.g., Academic Pressure and Study Satisfaction). |
| **Support Vector Machine** | Achieves the top overall accuracy (**84.38%**), highest F1 score (**0.8684**), and highest Matthews Correlation Coefficient (**0.6766**). The Radial Basis Function (RBF) kernel accurately constructs a non-linear margin that accommodates subtle feature interactions. |

### Overall Winner for Your Dataset:

> **🏆 Winner: Support Vector Machine (SVM) / Logistic Regression**
> 
> **Justification:**
> 1. **Highest MCC (0.6766) and Accuracy (84.38%):** Matthews Correlation Coefficient is the most robust metric for binary classification as it evaluates all four quadrants of the confusion matrix (TP, TN, FP, FN). SVM leads all models with an MCC of 0.6766.
> 2. **Optimal Precision-Recall Balance:** In mental health screening, both false alarms (low precision) and missed diagnoses (low recall) carry clinical costs. SVM provides an ideal equilibrium with **85.68% Precision** and **88.04% Recall** ($F_1 = 0.8684$).
> 3. **High Discriminative Power:** With a ROC-AUC of **0.9157** (and Logistic Regression at **0.9184**), both models exhibit exceptional class separability across varying decision thresholds.

---

## 6. Repository Structure

```text
student-mental-health-classifier/
│
├── app.py                      # Interactive Streamlit Web Application
├── requirements.txt            # Reproducible python dependencies
├── run.sh                      # One-step setup & launch (macOS/Linux/Git Bash/WSL)
├── run.bat                     # One-step setup & launch (Windows cmd/PowerShell)
├── README.md                   # Full assignment documentation & analysis
├── Assignment2.md              # Assignment specification & marking rubric
│
├── Datasets/
│   ├── Student_Depression_Dataset.csv   # Original dataset (27,901 rows)
│   └── test_data.csv                    # 5,581 sample holdout test set with ground truth
│
├── model/                      # Serialized pre-trained artifacts
│   ├── preprocessor.joblib     # Fitted ColumnTransformer (Scaler + OneHotEncoder)
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   ├── metadata.json           # Feature schema and target definition
│   ├── metrics_summary.csv     # Tabular metric comparison
│   └── detailed_results.json   # Full confusion matrices & classification reports
│
├── notebooks/
│   └── student_depression_analysis.ipynb # End-to-end training notebook for BITS Lab
│
└── src/
    └── train.py                # Standalone training & evaluation pipeline script
```

---

## 7. Streamlit Web Application Features

The deployed Streamlit application includes all mandatory and bonus features:
- **a. Test Dataset Upload (CSV):** Allows users to upload their own test CSV or defaults to the included `Datasets/test_data.csv` (5,581 samples).
- **b. Model Selection Dropdown:** Seamless switching between all 5 classifiers or a consolidated "Compare All 5 Models" mode.
- **c. Evaluation Metrics Dashboard:** Displays Accuracy, ROC-AUC, Precision, Recall, F1, and MCC via modern KPI cards.
- **d. Visual Confusion Matrix & ROC Curves:** Interactive Plotly heatmaps and combined ROC curves.
- **e. Full Classification Report:** Detailed per-class precision, recall, and support breakdown.
- **f. Interactive Risk Simulator:** Real-time single-student profile assessment using form inputs and what-if sliders.

---

## 8. Execution Instructions (BITS Virtual Lab & Local)

### A. Local Setup & Training
```bash
# 1. Clone repository
git clone https://github.com/bhoomikakodi/student-mental-health-classifier.git
cd student-mental-health-classifier

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows (cmd): venv\Scripts\activate | PowerShell: venv\Scripts\Activate.ps1

# 3. Install dependencies (always install from requirements.txt inside the venv)
pip install -r requirements.txt
#    (Notebook users: register the venv as a kernel and select it in Jupyter/VS Code)
#    pip install ipykernel && python -m ipykernel install --user --name student-mental-health-venv

# 4. Train all models and export artifacts
python src/train.py

# 5. Launch interactive Streamlit application
streamlit run app.py
```

**One-step alternative:** run all of steps 2–5 above with a helper script from the project root. Each script creates the `venv` (if missing), activates it, installs `requirements.txt`, registers the Jupyter kernel, trains the models and launches the app:
- **macOS / Linux / Git Bash / WSL:** `./run.sh` (auto-detects `venv/bin/activate` or `venv/Scripts/activate`)
- **Windows `cmd` / PowerShell:** `run.bat` (uses `venv\Scripts\activate`; in PowerShell use `.\run.bat`)

**Running the notebook:** open `notebooks/student_depression_analysis.ipynb` and select the **Python (student-mental-health venv)** kernel (or `venv/bin/python`). Its first cell runs `%pip install -r ../requirements.txt` inside that venv.

### B. Deployment to Streamlit Community Cloud
1. Push this repository to GitHub.
2. Navigate to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub account and click **"New App"**.
4. Select the repository, set branch to `main`, and main file path to `app.py`.
5. Click **Deploy**.

---

## 9. Submission Checklist

- [x] Dataset meets criteria ($\ge 12$ features, $\ge 500$ instances — actual: 16 features, 27,901 rows).
- [x] All 5 classification algorithms trained, saved, and evaluated.
- [x] All 6 evaluation metrics computed and compared (Accuracy, AUC, Precision, Recall, F1, MCC).
- [x] Streamlit app (`app.py`) loads models without retraining and supports test CSV upload.
- [x] Clean folder structure with `model/`, `Datasets/`, `requirements.txt`, `src/`, `notebooks/`.
- [x] Markdown documentation and comparison tables strictly follow grading guidelines.
