"""
Public Health Modelling – SIR / SEIR Epidemic Model
=====================================================
Implements compartmental epidemic models (SIR and SEIR) and applies
them to simulate disease spread in a closed population.  Includes
reproduction number (R0) estimation and intervention scenario comparison.
"""

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Model parameters (default: influenza-like illness)
# ---------------------------------------------------------------------------
POPULATION = 100_000        # Total population N
INITIAL_INFECTED = 10       # I(0)
INITIAL_EXPOSED = 50        # E(0)  – SEIR only
BETA = 0.30                 # Transmission rate (contacts per day × probability)
GAMMA = 0.10                # Recovery rate (1/duration_of_infection)
SIGMA = 0.20                # Progression rate from exposed → infectious (1/incubation)
T_MAX = 180                 # Simulation horizon (days)


# ---------------------------------------------------------------------------
# SIR Model
# ---------------------------------------------------------------------------

def sir_model(t, y, beta, gamma, N):
    """ODE system for the SIR compartmental model."""
    S, I, R = y
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    return [dS, dI, dR]


def run_sir(beta=BETA, gamma=GAMMA, N=POPULATION,
            I0=INITIAL_INFECTED, t_max=T_MAX):
    """
    Solve the SIR model and return a dict with time and compartment arrays.
    """
    S0 = N - I0
    y0 = [S0, I0, 0]  # S, I, R initial conditions
    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, t_max + 1)

    sol = solve_ivp(sir_model, t_span, y0, args=(beta, gamma, N),
                    t_eval=t_eval, method="RK45", dense_output=True)
    return {
        "t": sol.t,
        "S": sol.y[0],
        "I": sol.y[1],
        "R": sol.y[2],
    }


# ---------------------------------------------------------------------------
# SEIR Model
# ---------------------------------------------------------------------------

def seir_model(t, y, beta, sigma, gamma, N):
    """ODE system for the SEIR compartmental model."""
    S, E, I, R = y
    dS = -beta * S * I / N
    dE = beta * S * I / N - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I
    return [dS, dE, dI, dR]


def run_seir(beta=BETA, sigma=SIGMA, gamma=GAMMA, N=POPULATION,
             I0=INITIAL_INFECTED, E0=INITIAL_EXPOSED, t_max=T_MAX):
    """
    Solve the SEIR model and return a dict with time and compartment arrays.
    """
    S0 = N - I0 - E0
    R0_init = 0
    y0 = [S0, E0, I0, R0_init]
    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, t_max + 1)

    sol = solve_ivp(seir_model, t_span, y0, args=(beta, sigma, gamma, N),
                    t_eval=t_eval, method="RK45", dense_output=True)
    return {
        "t": sol.t,
        "S": sol.y[0],
        "E": sol.y[1],
        "I": sol.y[2],
        "R": sol.y[3],
    }


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------

def basic_reproduction_number(beta, gamma):
    """Compute R0 = beta / gamma."""
    return round(beta / gamma, 4)


def effective_reproduction_number(beta, gamma, S, N):
    """Compute Rt = R0 × (S / N) at each time step."""
    R0 = beta / gamma
    return R0 * S / N


def peak_infected(result):
    """Return (peak_day, peak_count) for the infected compartment."""
    peak_day = int(np.argmax(result["I"]))
    peak_count = int(round(result["I"][peak_day]))
    return peak_day, peak_count


def total_attack_rate(result, N=POPULATION):
    """Fraction of population ever infected (final R / N)."""
    return round(float(result["R"][-1]) / N, 4)


# ---------------------------------------------------------------------------
# Intervention comparison
# ---------------------------------------------------------------------------

def compare_interventions(N=POPULATION, t_max=T_MAX):
    """
    Compare baseline vs. two intervention scenarios that reduce beta.
    Returns a list of dicts with scenario label and key metrics.
    """
    scenarios = [
        {"label": "Baseline (no intervention)", "beta": 0.30},
        {"label": "Moderate intervention (30% beta reduction)", "beta": 0.21},
        {"label": "Strong intervention (60% beta reduction)", "beta": 0.12},
    ]
    results = []
    for sc in scenarios:
        r = run_sir(beta=sc["beta"], N=N, t_max=t_max)
        pd, pc = peak_infected(r)
        results.append(
            {
                "scenario": sc["label"],
                "R0": basic_reproduction_number(sc["beta"], GAMMA),
                "peak_day": pd,
                "peak_infected": pc,
                "attack_rate_pct": round(total_attack_rate(r, N) * 100, 1),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("Public Health Modelling – SIR / SEIR Epidemic Simulation")
    print("=" * 65)

    R0 = basic_reproduction_number(BETA, GAMMA)
    print(f"\nModel parameters:")
    print(f"  Population (N)    : {POPULATION:,}")
    print(f"  Beta              : {BETA}")
    print(f"  Gamma             : {GAMMA}")
    print(f"  Sigma (SEIR only) : {SIGMA}")
    print(f"  Basic R0          : {R0}")

    print("\n--- SIR Model ---")
    sir = run_sir()
    pd_sir, pc_sir = peak_infected(sir)
    print(f"  Peak infected   : {pc_sir:,} on day {pd_sir}")
    print(f"  Attack rate     : {total_attack_rate(sir)*100:.1f}% of population")
    print(f"  Final susceptible: {int(sir['S'][-1]):,}")
    print(f"  Final recovered  : {int(sir['R'][-1]):,}")

    print("\n--- SEIR Model ---")
    seir = run_seir()
    pd_seir, pc_seir = peak_infected(seir)
    print(f"  Peak infected   : {pc_seir:,} on day {pd_seir}")
    print(f"  Attack rate     : {total_attack_rate(seir)*100:.1f}% of population")
    print(f"  Final susceptible: {int(seir['S'][-1]):,}")
    print(f"  Final recovered  : {int(seir['R'][-1]):,}")

    print("\n--- Intervention Comparison (SIR) ---")
    header = f"{'Scenario':<45} {'R0':>5} {'Peak Day':>10} {'Peak Infected':>15} {'Attack Rate':>12}"
    print(header)
    print("-" * len(header))
    for row in compare_interventions():
        print(
            f"{row['scenario']:<45} {row['R0']:>5} {row['peak_day']:>10} "
            f"{row['peak_infected']:>15,} {row['attack_rate_pct']:>11.1f}%"
        )

    print("\nSimulation complete.")
    return sir, seir


if __name__ == "__main__":
    main()
