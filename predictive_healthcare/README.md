# Predictive Healthcare Systems – Hospital Readmission Prediction

Trains and evaluates machine-learning models to predict 30-day hospital
readmission risk from synthetic patient features (demographics,
comorbidities, prior utilisation, lab values).

## Overview

| Item | Detail |
|------|--------|
| Sample size | 1,000 patients |
| Task | Binary classification (30-day readmission) |
| Features | 14 clinical & demographic features |
| Models | Logistic Regression, Random Forest, Gradient Boosting |

## Features

| Feature | Description |
|---------|-------------|
| age | Patient age |
| gender | Male / Female |
| insurance | Insurance type |
| hypertension / diabetes / heart_disease / obesity | Comorbidity flags |
| comorbidity_count | Total comorbidities |
| prior_admissions | # admissions in the past year |
| los_days | Length of current stay (days) |
| er_visits_prior_year | ER visits in prior 12 months |
| hba1c | Glycated haemoglobin (%) |
| sodium / creatinine | Lab values |

## Evaluation

- 5-fold stratified cross-validation (ROC-AUC)
- Hold-out test set: ROC-AUC, Brier score, confusion matrix
- Feature importance (Gradient Boosting)

## Usage

```bash
pip install numpy pandas scikit-learn
python hospital_readmission_prediction.py
```
