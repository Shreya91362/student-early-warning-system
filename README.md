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
   
3. Run the data generator to create the dataset.  
4. Execute the training notebook or script to train the model.  
5. Launch the dashboard notebook to perform predictions.  

---

## Note on Model Files
Trained model artifacts are excluded from this repository due to size limitations.

To regenerate the models, run the training notebook or scripts provided in the project.

---

## Project Timeline

Week 1  - Problem Definition and Proposal  
Week 2  - Data Generation  
Week 3-4 - Model Training and Evaluation  
Week 5  - Retraining Pipeline Development  
Week 6  - Dashboard Implementation  
Week 7  - Testing and Optimization  

---

## Expected Outcome
The project delivers a deployment-ready predictive system capable of identifying students at academic risk with high reliability.

By integrating training, retraining, analytics, and inference into a unified architecture, the solution demonstrates the practical application of machine learning in an educational context.


