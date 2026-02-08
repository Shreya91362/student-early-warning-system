# Early Warning System for Student Academic Risk Prediction

## Problem Statement
Universities often face challenges in identifying students who are at academic risk before their performance declines significantly. Traditional monitoring methods are largely manual, reactive, and time-consuming.

This project develops a machine learning-based Early Warning System that predicts student risk levels using academic performance indicators. The system supports proactive intervention and enables data-driven academic decision-making.

---

## Solution Overview
The proposed solution applies traditional machine learning algorithms to analyze student performance data and classify students into risk categories.

A synthetic data generator is implemented to simulate realistic academic records, ensuring continuous data availability without relying on sensitive institutional data.

The following models were trained and evaluated:

- Logistic Regression  
- Decision Tree  
- Random Forest  

After comparative evaluation, Random Forest was selected as the final deployment model due to its robustness and strong generalization capability.

The system integrates training, evaluation, prediction, and periodic retraining into a unified pipeline.

---

## System Architecture

Synthetic Data Generator  
        ↓  
Model Training and Evaluation  
        ↓  
Saved Model  
        ↓  
Interactive Dashboard  
        ↓  
Real-Time Predictions  
        ↓  
Periodic Retraining  

This architecture ensures the system remains adaptable to evolving academic patterns while maintaining predictive reliability.

---

## Features

- Synthetic academic data generation  
- Training of multiple machine learning models  
- Model performance evaluation  
- Feature importance analysis for interpretability  
- Interactive dashboard for analytics and prediction  
- Real-time risk classification with confidence scores  
- Structured retraining pipeline  
- Reproducible project setup  

---

## Tech Stack

- Python  
- Scikit-learn  
- Pandas  
- NumPy  
- Matplotlib  
- Joblib  
- ipywidgets  

---

## Model Lifecycle and Retraining Strategy
The system follows a structured model lifecycle to maintain predictive performance.

- New student data can be appended periodically.  
- The training pipeline can be re-executed to retrain the model.  
- Retraining is recommended at the end of each academic term or when performance metrics decline.  
- This approach helps mitigate model drift and ensures long-term reliability.  

---

## Dashboard and Predictions
An interactive dashboard was developed within the notebook environment to visualize the distribution of students across risk categories and enable prediction functionality.

Users can input student parameters such as attendance, GPA, study hours, and assessment scores to generate immediate risk predictions along with confidence levels.

---

## Repository Structure

data_generator/   - Synthetic data creation  
notebooks/        - Model training and retraining  
dashboard/        - Interactive prediction dashboard  
models/           - Documentation for model artifacts  
docs/             - Project proposal  

---

## How to Run the Project

1. Clone the repository.  
2. Install dependencies:

