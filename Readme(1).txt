Deep Learning for Health Care (DLHC)
Project Overview
This project focuses on building a machine learning pipeline to analyze patient ICU data using the MIMIC-III clinical dataset.
The goal is to create a clean, structured patient cohort and develop predictive models for in-hospital and ICU mortality using demographic, lab, and clinical data.
Dataset
Source: MIMIC-III (Medical Information Mart for Intensive Care)
Tables used:
ADMISSIONS.csv
PATIENTS.csv
ICUSTAYS.csv
LABEVENTS.csv
Folder Location: DLHC/DATASET/
The dataset folder is large and is excluded from GitHub using a .gitignore file.
Data Preparation
Merged multiple tables (ADMISSIONS, PATIENTS, ICUSTAYS, and lab features).
Computed patient age, ICU stay duration, and mortality flags.
Extracted key lab features such as:
Glucose
Hemoglobin
Lactate
Potassium
Sodium
WBC (White Blood Cells)
Final cohort shape: (283, 17) — ready for ML/DL modeling.
Modeling
Machine Learning Models Implemented:
Model: Logistic Regression
Accuracy: 82.4%
Highlights: Good baseline; high precision for survivors
Model: Random Forest
Accuracy: 78.9%
Highlights: Robust but slightly lower recall for mortality
Tech Stack
Language: Python
Environment: Anaconda (conda environment: mimic_icu)
Tools: pandas, numpy, scikit-learn, matplotlib, seaborn, torch
Editor: Visual Studio Code (Jupyter Notebook integrated)
Version Control: Git and GitHub integration directly from VSCode
Future Enhancements
Apply SMOTE or class weighting to handle class imbalance.
Perform feature importance analysis.
Build a Neural Network classifier (PyTorch) for mortality prediction.
Add visual analytics (age distribution, lab trends, etc.).
Compare multiple ensemble models (XGBoost, LightGBM).
Workflow Summary
Work inside VSCode (Jupyter notebook view).
Commit regularly through the Source Control tab.
Sync changes to GitHub using the "Sync Changes" button.
The .gitignore file ensures that large datasets are not pushed to GitHub.
Repository Structure
DLHC/
DATASET/ - Local dataset folder (excluded from GitHub)
cohort_building.ipynb - Data cleaning and cohort creation notebook
modeling.ipynb - ML models and evaluations
README.md - Project documentation (this file)
.gitignore - Ignores DATASET and temporary files
Author
Amarnath Reddy N
B.Tech in Artificial Intelligence, Mahindra University
Passionate about AI/ML and healthcare innovation.