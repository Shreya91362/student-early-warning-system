# ════════════════════════════════════════════════════════════════════════════
# Retraining Script — Student Early Warning System
# Run this when 50+ new students are added or accuracy drops
# ════════════════════════════════════════════════════════════════════════════

import sqlite3
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from datetime import datetime

DB_PATH    = 'student_risk.db'
MODEL_PATH = 'best_model.pkl'

FEATURE_COLS = [
    'year', 'cgpa', 'previous_failures', 'assignment_completion',
    'attendance_rate', 'unexcused_absences', 'behavioral_incidents',
    'days_since_incident', 'scholarship', 'extracurricular',
    'hostel_resident', 'part_time_job'
]

def load_data():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    q = '''
    SELECT s.year,
        CAST(ar.cgpa AS REAL) as cgpa,
        CAST(ar.previous_failures AS INTEGER) as previous_failures,
        CAST(ar.assignment_completion AS REAL) as assignment_completion,
        CAST(COALESCE((SELECT ROUND(SUM(CASE WHEN a.status='Present'
            THEN 1.0 ELSE 0 END)/30,3)
            FROM attendance a WHERE a.student_id=s.student_id),0.85) AS REAL)
            as attendance_rate,
        CAST(COALESCE((SELECT SUM(a.unexcused) FROM attendance a
            WHERE a.student_id=s.student_id),0) AS INTEGER) as unexcused_absences,
        CAST(COALESCE(bi.total_incidents,0) AS INTEGER) as behavioral_incidents,
        CAST(COALESCE(bi.days_since_last,365) AS INTEGER) as days_since_incident,
        CAST(COALESCE(sd.scholarship,0) AS INTEGER) as scholarship,
        CAST(COALESCE(sd.extracurricular,0) AS INTEGER) as extracurricular,
        CAST(COALESCE(sd.hostel_resident,0) AS INTEGER) as hostel_resident,
        CAST(COALESCE(sd.part_time_job,0) AS INTEGER) as part_time_job
    FROM students s
    LEFT JOIN academic_records ar ON s.student_id=ar.student_id
    LEFT JOIN behavioral_incidents bi ON s.student_id=bi.student_id
    LEFT JOIN student_demographics sd ON s.student_id=sd.student_id
    WHERE ar.cgpa IS NOT NULL
    '''
    df = pd.read_sql_query(q, conn)
    conn.close()
    return df

def create_labels(df):
    cgpa      = df['cgpa']
    att       = df['attendance_rate']
    comp      = df['assignment_completion']
    incidents = df['behavioral_incidents']
    failures  = df['previous_failures']

    critical = (cgpa < 5.5) | (att < 0.82)
    w1 = (cgpa < 6.0).astype(int)
    w2 = (att < 0.86).astype(int)
    w3 = (comp < 0.65).astype(int)
    w4 = (incidents >= 2).astype(int)
    w5 = (failures >= 2).astype(int)
    warning_count = w1 + w2 + w3 + w4 + w5

    base_labels = (critical | (warning_count >= 2)).astype(int)
    noise_mask  = np.random.random(len(df)) < 0.15
    labels      = base_labels.copy()
    labels[noise_mask] = 1 - labels[noise_mask]
    return labels

def retrain():
    print("=" * 60)
    print("RETRAINING — Student Early Warning System")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    df = load_data()
    print(f"Loaded {len(df)} students")

    df['at_risk'] = create_labels(df)
    print(f"At-Risk: {df['at_risk'].sum()} | Safe: {(df['at_risk']==0).sum()}")

    X = df[FEATURE_COLS].fillna(0)
    y = df['at_risk']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=100, random_state=42,
        max_depth=5, min_samples_leaf=5,
        max_features='sqrt', class_weight='balanced'
    )
    model.fit(X_train_scaled, y_train)

    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    test_acc  = accuracy_score(y_test, model.predict(X_test_scaled))

    print(f"CV Accuracy  : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"Test Accuracy: {test_acc:.3f}")

    pkg = {
        'model':            model,
        'scaler':           scaler,
        'feature_columns':  FEATURE_COLS,
        'model_type':       'Random Forest',
        'training_date':    datetime.now().strftime('%Y-%m-%d %H:%M'),
        'training_samples': len(X_train),
        'performance':      {'Accuracy': test_acc},
        'test_y_true':      list(y_test.values),
        'test_y_pred':      list(model.predict(X_test_scaled)),
    }
    joblib.dump(pkg, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print("Retraining complete!")

if __name__ == '__main__':
    retrain()
