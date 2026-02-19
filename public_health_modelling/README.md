# Public Health Modelling – SIR / SEIR Epidemic Model

Implements compartmental epidemic models (SIR and SEIR) to simulate
disease spread in a closed population.  Includes R₀ estimation and
intervention scenario comparisons.

## Overview

| Item | Detail |
|------|--------|
| Models | SIR, SEIR |
| Population | 100,000 |
| Default pathogen | Influenza-like illness |
| Simulation horizon | 180 days |

## Models

- **SIR** – Susceptible → Infected → Recovered
- **SEIR** – Susceptible → Exposed → Infected → Recovered

## Analysis

- Basic reproduction number R₀ = β / γ
- Effective reproduction number Rₜ over time
- Peak infection day and magnitude
- Population attack rate
- Intervention scenarios comparing 0%, 30%, and 60% reductions in transmission rate

## Usage

```bash
pip install numpy scipy
python sir_seir_epidemic_model.py
```
