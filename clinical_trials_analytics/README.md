# Clinical Trials Analytics

Analyzes synthetic clinical trial data to evaluate treatment efficacy,
patient enrollment trends, adverse events, and trial outcome statistics.

## Overview

| Item | Detail |
|------|--------|
| Trial design | Randomized controlled trial (RCT) |
| Arms | Treatment vs. Control (1:1 allocation) |
| Sample size | 300 patients |
| Primary endpoint | Continuous efficacy score (0–100) |
| Duration | 24 weeks |

## Analysis

- **Descriptive statistics** – per-arm summaries (age, baseline/endpoint scores, AE rate)
- **Efficacy test** – Independent-samples t-test on endpoint scores
- **Adverse event analysis** – Chi-square test for independence
- **Enrollment trends** – Cumulative weekly enrollment by arm

## Usage

```bash
pip install numpy pandas scipy
python clinical_trials_analysis.py
```
