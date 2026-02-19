# Clinical & Healthcare Data Science Projects

A collection of Clinical & Healthcare Data Science projects focused on
clinical trials analytics, patient demographics, public health modelling,
and predictive healthcare systems using Python and Machine Learning.

---

## Projects

### 1. [Clinical Trials Analytics](clinical_trials_analytics/)

Analyzes a synthetic randomized controlled trial (RCT) dataset to evaluate
treatment efficacy, patient enrollment trends, adverse events, and trial
outcome statistics.

**Key techniques:** descriptive statistics, independent t-test, chi-square test,
enrollment trend analysis.

### 2. [Patient Demographics Analysis](patient_demographics/)

Explores a synthetic patient registry to surface age distributions, gender
balance, ethnicity breakdowns, comorbidity profiles, and the relationship
between demographics and 30-day readmission outcomes.

**Key techniques:** group-level summaries, prevalence analysis, one-way ANOVA.

### 3. [Public Health Modelling – SIR / SEIR](public_health_modelling/)

Implements SIR and SEIR compartmental epidemic models to simulate disease
spread in a closed population.  Includes R₀ estimation, effective
reproduction number tracking, and intervention scenario comparisons.

**Key techniques:** ODE-based epidemic modelling (scipy `solve_ivp`), R₀ / Rₜ
computation, scenario analysis.

### 4. [Predictive Healthcare Systems – Readmission Prediction](predictive_healthcare/)

Trains and evaluates machine-learning classifiers (Logistic Regression,
Random Forest, Gradient Boosting) to predict 30-day hospital readmission
risk from demographics, comorbidities, prior utilisation, and lab values.

**Key techniques:** scikit-learn pipelines, stratified k-fold CV, ROC-AUC,
Brier score, feature importance.

---

## Repository Structure

```
.
├── clinical_trials_analytics/
│   ├── clinical_trials_analysis.py
│   └── README.md
├── patient_demographics/
│   ├── patient_demographics_analysis.py
│   └── README.md
├── public_health_modelling/
│   ├── sir_seir_epidemic_model.py
│   └── README.md
├── predictive_healthcare/
│   ├── hospital_readmission_prediction.py
│   └── README.md
├── requirements.txt
└── README.md
```

## Requirements

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---------|---------|
| numpy | Numerical computing |
| pandas | Data manipulation |
| scipy | Statistical tests & ODE solver |
| scikit-learn | Machine learning models & evaluation |

## Running the Projects

```bash
# Clinical Trials Analytics
python clinical_trials_analytics/clinical_trials_analysis.py

# Patient Demographics Analysis
python patient_demographics/patient_demographics_analysis.py

# Public Health Modelling
python public_health_modelling/sir_seir_epidemic_model.py

# Predictive Healthcare – Readmission Prediction
python predictive_healthcare/hospital_readmission_prediction.py
```