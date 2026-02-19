# Patient Demographics Analysis

Analyzes synthetic patient demographic data to surface age distributions,
gender balance, ethnicity breakdowns, comorbidity profiles, and the
relationship between demographics and health outcomes.

## Overview

| Item | Detail |
|------|--------|
| Sample size | 500 patients |
| Demographics | Age, gender, ethnicity, insurance type |
| Comorbidities | Hypertension, Diabetes, Obesity, Heart Disease, Asthma |
| Outcome | 30-day hospital readmission |

## Analysis

- **Age group distribution** – binned into clinical age ranges
- **Gender & ethnicity breakdowns** – counts and percentages
- **Comorbidity prevalence** – population-level prevalence rates
- **Readmission rates by age group** – outcome stratification
- **One-way ANOVA** – test for ethnicity effect on readmission

## Usage

```bash
pip install numpy pandas scipy
python patient_demographics_analysis.py
```
