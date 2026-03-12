import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def seir_euler(beta, sigma, gamma, S0, E0, I0, R0, timepoints, N):

    # Initialize S, E, I, and R as empty arrays
    S = np.zeros(len(timepoints))
    E = np.zeros(len(timepoints))
    I = np.zeros(len(timepoints))
    R = np.zeros(len(timepoints))

    # Set first item in each list equal to initial values
    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0

    # Step size (assumes evenly spaced timepoints)
    dt = timepoints[1] - timepoints[0]

    # For each timepoint in timepoints:
    for idx in range(len(timepoints) - 1):
        # Calculate the four derivatives at current timepoint
        dS = -beta * S[idx] * I[idx] / N
        dE =  beta * S[idx] * I[idx] / N - sigma * E[idx]
        dI =  sigma * E[idx] - gamma * I[idx]
        dR =  gamma * I[idx]

        # Calculate S, E, I, R at next timepoint using Euler's method
        # y(t + dt) = y(t) + dt * y'(t)
        S[idx + 1] = S[idx] + dt * dS
        E[idx + 1] = E[idx] + dt * dE
        I[idx + 1] = I[idx] + dt * dI
        R[idx + 1] = R[idx] + dt * dR

    return S, E, I, R

N       = 20_000   # assumed total population
I0      = 1
E0      = 0
R0_init = 0
S0      = N - I0 - E0 - R0_init



# Load data 
df = pd.read_csv('Data\mystery_virus_daily_active_counts_RELEASE#2.csv')
data_days = df['day'].values
data_I    = df['active reported daily cases'].values

timepoints = np.arange(1, data_days[-1] + 1, 1)  # one timepoint per day

# Grid search
best_SSE    = np.inf
best_params = None

for beta in np.arange(0.40, 0.65, 0.01):
    for sigma in np.arange(0.30, 0.50, 0.01):
        for gamma in np.arange(0.15, 0.30, 0.01):

            S, E, I, R = seir_euler(beta, sigma, gamma,
                                    S0, E0, I0, R0_init,
                                    timepoints, N)

            # Compare model I(t) to observed data at matching days
            model_I_at_data = I[data_days - 1]   # data_days are 1-indexed
            SSE = np.sum((model_I_at_data - data_I) ** 2)

            if SSE < best_SSE:
                best_SSE    = SSE
                best_params = (beta, sigma, gamma)

beta, sigma, gamma = best_params
print(f"Best-fit parameters:")
print(f"  beta  = {beta:.3f}")
print(f"  sigma = {sigma:.3f}  (incubation period ≈ {1/sigma:.1f} days)")
print(f"  gamma = {gamma:.3f}  (infectious period ≈ {1/gamma:.1f} days)")
print(f"  R0    = beta/gamma = {beta/gamma:.2f}")
print(f"  SSE   = {best_SSE:.1f}")

# Run model with best params
t_extended = np.arange(1, 201, 1)
S, E, I, R = seir_euler(beta, sigma, gamma, S0, E0, I0, R0_init, t_extended, N)

peak_day = t_extended[np.argmax(I)]
peak_val = np.max(I)
print(f"\nPredicted peak: {peak_val:.0f} active cases on day {int(peak_day)}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: model vs data over the data window
ax = axes[0]
ax.plot(t_extended[:data_days[-1]], I[:data_days[-1]],
        color='steelblue', linewidth=2, label='Model I(t)')
ax.scatter(data_days, data_I,
           color='crimson', s=25, zorder=5, label='Observed data (Release 2)')
ax.set_xlabel('Day')
ax.set_ylabel('Active infectious cases')
ax.set_title('SEIR Fit to Mystery Virus Data')
ax.legend()
ax.grid(alpha=0.3)

plt.show()

# Load full dataset (Release 3, days 1-120)
df = pd.read_csv('Data\mystery_virus_daily_active_counts_RELEASE#3.csv')
data_days = df['day'].values
data_I    = df['active reported daily cases'].values

# True peak from data 
true_peak_val = np.max(data_I)
true_peak_day = data_days[np.argmax(data_I)]
print(f"True peak:  {true_peak_val} cases on day {true_peak_day}")

# Model prediction (best-fit params from Release 2 grid search) 
N     = 20_000
beta  = 0.490
sigma = 0.410
gamma = 0.220

I0      = 1
E0      = 0
R0_init = 0
S0      = N - I0 - E0 - R0_init

t_extended = np.arange(1, 201, 1)   # run out to day 200 to capture full peak
S, E, I, R = seir_euler(beta, sigma, gamma, S0, E0, I0, R0_init, t_extended, N)

model_peak_val = np.max(I)
model_peak_day = t_extended[np.argmax(I)]
print(f"Model peak: {model_peak_val:.1f} cases on day {model_peak_day:.0f}")

# True % relative error
# Formula: % et = (true_value - approximation) / true_value * 100
pct_err_y = (true_peak_val - model_peak_val) / true_peak_val * 100
pct_err_x = (true_peak_day - model_peak_day) / true_peak_day * 100

print(f"\nTrue % relative error — peak cases (y): {pct_err_y:.2f}%")
print(f"True % relative error — peak day   (x): {pct_err_x:.2f}%")