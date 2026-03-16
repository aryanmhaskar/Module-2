import numpy as np                     
import matplotlib.pyplot as plt        
import pandas as pd                    
from scipy.optimize import minimize    


# SEIR model function using Euler's method

def seir_euler(beta, sigma, gamma, S0, E0, I0, R0, timepoints, N):
    """
        beta  - transmission rate
        sigma - rate exposed
        gamma - recovery rate
        S0, E0, I0, R0 - initial populations
        timepoints - array of time steps (days)
        N - total population
    """
    # arrays for SEIR
    S = np.zeros(len(timepoints))
    E = np.zeros(len(timepoints))
    I = np.zeros(len(timepoints))
    R = np.zeros(len(timepoints))

    # set initial values
    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0

    # time step
    dt = timepoints[1] - timepoints[0]

    # Euler's method
    for idx in range(len(timepoints) - 1):
        # derivatives
        dS = -beta * S[idx] * I[idx] / N         # Susceptibles becoming exposed
        dE = beta * S[idx] * I[idx] / N - sigma * E[idx]  # Exposed becoming infectious
        dI = sigma * E[idx] - gamma * I[idx]     # Infectious recovering
        dR = gamma * I[idx]                       # Recovered individuals

        # Update compartments for next time step
        S[idx + 1] = S[idx] + dt * dS
        E[idx + 1] = E[idx] + dt * dE
        I[idx + 1] = I[idx] + dt * dI
        R[idx + 1] = R[idx] + dt * dR

    return S, E, I, R


# Load observed data from CSV (Release 3)

df = pd.read_csv(r'Data\mystery_virus_daily_active_counts_RELEASE#3.csv')
data_days = df['day'].values                              # day numbers
data_I    = df['active reported daily cases'].values      # observed active infections


# inital SEIR parameters for first 70 days

N       = 20000      # total population
beta    = 0.490      # transmission rate
sigma   = 0.410      # exposed infectious rate
gamma   = 0.220      # recovery rate
I0      = 1          # initial infectious
E0      = 0          # initial exposed
R0_init = 0          # initial recovered
S0      = N - I0 - E0 - R0_init  # initial susceptible population


# runs SEIR for days 1-70 to get the state at day 70

t1_70 = np.arange(1, 71)  # Days 1 to 70
S, E, I, R = seir_euler(beta, sigma, gamma, S0, E0, I0, R0_init, t1_70, N)


# updates SEIR parameters to match data from days 70-120

day_start = 70
day_end   = 120

# subset the observed data for days 70-120
days_post70 = data_days[(data_days >= day_start) & (data_days <= day_end)]
data_I_post70 = data_I[(data_days >= day_start) & (data_days <= day_end)]

# set initial state at day 70 from first SEIR run
S0_post70 = S[day_start - 1]
E0_post70 = E[day_start - 1]
I0_post70 = I[day_start - 1]
R0_post70 = R[day_start - 1]

t_post70 = np.arange(day_start, day_end + 1)  # timepoints for refit

# define sum-of-squares error function for fitting
def sse_seir(params):
    beta_try, sigma_try, gamma_try = params
    # Run SEIR with trial parameters
    S_temp, E_temp, I_temp, R_temp = seir_euler(
        beta_try, sigma_try, gamma_try,
        S0_post70, E0_post70, I0_post70, R0_post70,
        t_post70, N
    )
    # compute sum of squared differences between model and observed
    return np.sum((I_temp - data_I_post70) ** 2)

# optimize parameters to minimize SSE
res = minimize(
    sse_seir,
    x0=[beta, sigma, gamma],                  # start from previous best-fit
    bounds=[(0.1, 1), (0.1, 1), (0.05, 0.5)]  # parameter bounds
)


beta_new, sigma_new, gamma_new = res.x
print(f"Updated parameters for days 70-120: beta={beta_new:.3f}, sigma={sigma_new:.3f}, gamma={gamma_new:.3f}")


# run SEIR with updated parameters for days 70-120

S_post70, E_post70, I_post70, R_post70 = seir_euler(
    beta_new, sigma_new, gamma_new,
    S0_post70, E0_post70, I0_post70, R0_post70,
    t_post70, N
)


# vaccination intervention scenario

days_plot = np.arange(day_start, day_end + 1)

# copy the SEIR arrays so we can simulate vaccination separately
S_vac = S_post70.copy()
E_vac = E_post70.copy()
I_vac = I_post70.copy()
R_vac = R_post70.copy()

# apply vaccination on day 70
# used AI to explain how to code for the effects of vaccination
vaccinated = 2000
efficacy = 0.9
S_vac[0] -= vaccinated * efficacy   # remove effectively immunized from susceptible
R_vac[0] += vaccinated * efficacy   # add them to recovered (immune)

# continue SEIR simulation from day 70 with vaccination
for idx in range(len(days_plot) - 1):
    dS = -beta_new * S_vac[idx] * I_vac[idx] / N
    dE = beta_new * S_vac[idx] * I_vac[idx] / N - sigma_new * E_vac[idx]
    dI = sigma_new * E_vac[idx] - gamma_new * I_vac[idx]
    dR = gamma_new * I_vac[idx]

    # update compartments for next day
    S_vac[idx + 1] = S_vac[idx] + dS
    E_vac[idx + 1] = E_vac[idx] + dE
    I_vac[idx + 1] = I_vac[idx] + dI
    R_vac[idx + 1] = R_vac[idx] + dR


# plot results

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# left: SEIR model vs observed cases
axes[0].plot(days_plot, I_post70, label='Model Prediction', color='steelblue', linewidth=2)
axes[0].scatter(days_plot, data_I_post70, label='Observed Data', color='crimson')
axes[0].set_xlabel('Day')
axes[0].set_ylabel('Active cases')
axes[0].set_title('Model vs Observed: Days 70-120')
axes[0].legend()
axes[0].grid(alpha=0.3)

# right: SEIR model with and without vaccination
axes[1].plot(days_plot, I_post70, label='No Intervention', color='steelblue')
axes[1].plot(days_plot, I_vac, label='Vaccination Day 70', color='green')
axes[1].set_xlabel('Day')
axes[1].set_ylabel('Active cases')
axes[1].set_title('Effect of Vaccinating 2000 Students')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()