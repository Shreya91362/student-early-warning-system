# Student Early Warning System
## Vijaybhoomi University — AI-Powered Academic Risk Prediction

---

## Problem Statement
Universities face challenges identifying at-risk students before performance
declines significantly. Traditional monitoring is manual, reactive, and slow.

This system uses machine learning to predict student risk levels using academic
performance indicators, enabling proactive intervention before students fail
or drop out.

---

## Solution Overview
A machine learning pipeline analyzes student performance data and classifies
students into three risk levels — Low, Medium, and High.

A synthetic data generator simulates realistic academic records for
Vijaybhoomi University across five schools.

**7 models were trained and compared:**
- Logistic Regression
- Decision Tree
- Random Forest ← Selected Winner
- Gradient Boosting
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- XGBoost

**Random Forest** was selected as the final model based on accuracy and
explainability. Selection uses 5-fold cross-validation to avoid overfitting
on a single train/test split.

---

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy (CV) | 74.4% ± 3.3% |
| Precision | ~70% |
| Recall | ~72% |
| AUC-ROC | ~0.80 |

All metrics are evaluated on a held-out 20% test set — not the training data.
15% label noise is intentionally added to simulate real-world uncertainty and
prevent data leakage.

---

## System Architecture
```
Synthetic Data Generator (200 students, 5 schools)
            ↓
6-Table SQLite Database
            ↓
Feature Engineering + Domain-Rule Labels
            ↓
7-Model Training + Cross-Validation
            ↓
Best Model Saved (Random Forest)
            ↓
Streamlit Dashboard (7 tabs)
            ↓
Real-Time Risk Predictions + Alerts
            ↓
Periodic Retraining Pipeline
```

---

## Features
- Synthetic academic data generation for 5 Vijaybhoomi schools
- 6-table relational SQLite database
- Domain-rule based labels (no data leakage)
- 7 ML models compared with 15+ metrics
- 5-fold cross-validation for reliable model selection
- Feature importance analysis (CGPA, attendance top factors)
- Interactive Streamlit dashboard with 7 tabs:
  - Overview — risk distribution charts
  - Students — searchable student list
  - Analytics — CGPA vs risk scatter plots
  - Model Insights — confusion matrix, ROC, precision-recall
  - Alerts — high risk students needing intervention
  - Predict — quick risk check
  - Add Student — add new student with instant prediction
- Per-course attendance input with dynamic sliders
- Animated risk result card with factor pills
- Retraining pipeline with performance monitoring

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| ML | Scikit-learn, XGBoost |
| Dashboard | Streamlit, Plotly |
| Database | SQLite |
| Data | Pandas, NumPy |
| Deployment | Google Colab + Ngrok |
| Notebook | Jupyter (Google Colab) |

---

## University Schools Covered

| School | Programs |
|--------|----------|
| VSST | B.Tech CSE, B.Tech AI/ML |
| VSOD | B.Des Fashion, Product, Communication |
| VSOL | BA LLB, BBA LLB, LLB |
| JAGSOM | BBA, MBA |
| TSM | B.Mus Performance, Production, Composition, B.Tech Sound Engineering |

---

## Repository Structure
```
student-early-warning-system/
├── notebooks/
│   └── Student_Risk_Prediction.ipynb   # Main notebook (all cells)
├── dashboard/
│   └── dashboard_app.py                # Streamlit dashboard
├── data_generator/
│   └── generate_data.py                # Synthetic data generation
├── models/
│   └── README.md                       # Model artifacts info
├── docs/
│   └── proposal.pdf                    # Project proposal
├── scripts/
│   └── retrain.py                      # Retraining pipeline
├── requirements.txt                    # Dependencies
├── .gitignore                          # Excludes .db and .pkl files
└── README.md                           # This file
```

---

## How to Run

1. Open `notebooks/Student_Risk_Prediction.ipynb` in Google Colab
2. Run cells in order: Cell 1 → Cell 2 → Cell 3
3. Run Cell 5 → Cell 6 → Cell 7 (database setup)
4. Run Cell 9 (labels) → Cell 10 (train) → Cell 11 (save model)
5. Run Cell 12 (visualizations)
6. Get a free Ngrok token from ngrok.com
7. Paste token in Cell 26 and run to launch the live dashboard

---

## Label Design (No Data Leakage)

Labels are created using transparent domain-defined thresholds:

**Critical — any ONE triggers at-risk:**
- CGPA < 5.5
- Attendance < 82%

**Warning — any TWO trigger at-risk:**
- CGPA < 6.0
- Attendance < 86%
- Assignment completion < 65%
- Behavioral incidents ≥ 2
- Previous failures ≥ 2

15% random noise is added to prevent the model from memorising label rules.

---

## Retraining Strategy
- Retrain recommended every academic term
- Automatic trigger: 50+ new students OR 25% data growth
- Performance check runs automatically when new students are added
- If accuracy drops more than 5%, system flags for retraining

---

## Note on Model Files
Trained `.pkl` model files are excluded from this repository (size limits).
To regenerate, run the training notebook from Cell 9 onwards.
```



