# Heart Disease Classification Project

This repository contains a machine learning project that predicts whether or not a patient has heart disease based on their medical attributes. It utilizes a variety of data science and machine learning libraries to build, evaluate, and tune classification models.

## Project Overview
The goal of this project is to perform binary classification to predict the presence of heart disease using clinical parameters. The project follows a structured data science workflow:
1. **Data Exploration (EDA):** Visualizing feature distributions, class balances, and correlation matrix heatmaps.
2. **Model Training & Testing:** Splitting data into training and test sets to evaluate baseline performance.
3. **Hyperparameter Tuning:** Optimizing models using `KNeighborsClassifier` loops, `RandomizedSearchCV`, and `GridSearchCV`.
4. **Advanced Evaluation:** Utilizing ROC curves (`RocCurveDisplay`), Confusion Matrices, and cross-validated metrics (Accuracy, Precision, Recall, F1).

## Dataset
The dataset used is the classic **Heart Disease dataset** from the UCI Machine Learning Repository (`Heart-disease.csv`). It contains 303 patient records with 14 clinical features:
* `age`, `sex`, `cp` (chest pain type), `trestbps` (resting blood pressure), `chol` (serum cholesterol), etc.
* `target`: 1 = heart disease, 0 = no heart disease.

## Models Evaluated
* **Logistic Regression** (Optimized with GridSearchCV)
* **K-Nearest Neighbors** (Tuned manually across neighbors)
* **RandomForestClassifier** (Achieved top performance)

## Key Results
The final Random Forest model achieved a solid classification score, evaluated comprehensively across 5-fold cross-validation:
* **Accuracy:** ~81%
* **Precision:** ~84%
* **Recall:** ~85%
* **F1-Score:** ~84%

## Project Structure
```text
├── Heart_Disease_Analysis.ipynb  # Main Jupyter Notebook with all code
├── Heart-disease.csv             # Dataset used for training
├── README.md                     # Project documentation
└── requirements.txt              # Package dependencies
