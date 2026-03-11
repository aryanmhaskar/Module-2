#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
#%%
# Load the data
data = pd.read_csv('Data/mystery_virus_daily_active_counts_RELEASE#1.csv', parse_dates=['date'], header=0, index_col=None)
#%%
# We have day number, date, and active cases. We can use the day number and active cases to fit an exponential growth curve to estimate R0.
# Let's define the exponential growth function
def exponential_growth(t, r):
    return np.exp(r * t)

# Fit the exponential growth model to the data. 
# We'll use a handy function from scipy called CURVE_FIT that allows us to fit any given function to our data. 
popt, pcov = curve_fit(exponential_growth, data['day'], data['active reported daily cases'])
# We will fit the exponential growth function to the active cases data. HINT: Look up the documentation for curve_fit to see how to use it.

# Approximate R0 using this fit
r = popt[0]
print(f"Estimated R0: {r:.2f}")

# Add the fit as a line on top of your scatterplot.

plt.plot(data['day'], exponential_growth(data['day'], r), color='blue', label='Exponential Fit')
plt.plot(data['day'], data['active reported daily cases'], marker='o', color='red', label='Active Infections')

plt.title('DATA RELEASE #1: Day vs Active Infections')
plt.xlabel('Day')
plt.ylabel('Active Reported Daily Cases')
plt.legend()
plt.grid(True)

plt.show()


'''
What viruses have a similar R0? Use the viruses.html file to find a virus or 2 with a similar R0 and give a 1-2 sentence background of the diseases.
Our R0 estimate is 0.12, which is very small compared to most viruses. The closest viruses in viruses.html are Nipah, Hendra, and Hantavirus which have values of R0 = 0.5. These viruses 
Nipah virus is a zoonotic paramyxovirus first identified during a 1998–1999 outbreak in Malaysia and Singapore, causing severe encephalitis and respiratory disease in humans with high case fatality rates; it is transmitted from fruit bats (genus Pteropus) to humans directly or via intermediate hosts such as pigs.

Hendra virus is a closely related zoonotic paramyxovirus first detected in 1994 in Australia, transmitted from flying foxes (Pteropus bats) to horses and occasionally to humans, where it can cause severe respiratory and neurologic disease with high mortality.

Hantavirus refers to a group of rodent-borne viruses in the family Hantaviridae that can cause hantavirus pulmonary syndrome (HPS) in the Americas and hemorrhagic fever with renal syndrome (HFRS) in Europe and Asia, typically transmitted to humans through inhalation of aerosolized rodent excreta.

How accurate do you think your R0 estimate is?
Given that our R0 estimate is 0.12, which is much lower than typical values for infectious diseases, it is likely that our estimate is not accurate. We think it is too early in the disease process to get a good estimate as there is likely underreporting and misreporting of the data. We have little knowledge to when the infection truly started/its infection/transmission rates, etc. We predict that our estimates for R0 will become more accurate as the disease progresses throughout the population. 
'''

# TODO: Implement SEIR Modeling. Plot Euler's Method solutions for I(t) and comapre to the data. Guess beta, sigma, and gamma parameters and calculate SSE. 
''' INPUTS: timepoints, N, S0, E0, I0, R0, data
• Initialize a range for beta, sigma, and gamma
• Initialize an empty array of SSE
• Make arrays of values given each range for each parameter
• For b in beta
• For s in sigma
• For g in gamma
• Use the Euler method function you developed to calculate S, E, I, and R given those
parameters
• Calculate the SSE given the model results and the data and append this to the SSE
array
• Determine parameters corresponding to lowest SSE
• Return best_beta, best_sigma, and best_gamma and corresponding SSE'''


# INPUTS: beta, sigma, gamma, S0, E0, I0, R0, timepoints, N

def seir_euler(beta, sigma, gamma, S0, E0, I0, R0, timepoints, N):

    # time step
    dt = timepoints[1] - timepoints[0]

    # Initialize S, E, I, and R as empty arrays or lists
    S = np.zeros(len(timepoints))
    E = np.zeros(len(timepoints))
    I = np.zeros(len(timepoints))
    R = np.zeros(len(timepoints))

    # Set first item in each list equal to initial values S0, E0, I0, R0
    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0

    # For each timepoint in timepoints:
    for t in range(len(timepoints) - 1):

        # Calculate the four derivatives at timepoint
        dSdt = -beta * S[t] * I[t] / N
        dEdt = beta * S[t] * I[t] / N - sigma * E[t]
        dIdt = sigma * E[t] - gamma * I[t]
        dRdt = gamma * I[t]

        # Calculate S, E, I, and R at timepoint + 1 using Euler’s method
        S[t+1] = S[t] + dSdt * dt
        E[t+1] = E[t] + dEdt * dt
        I[t+1] = I[t] + dIdt * dt
        R[t+1] = R[t] + dRdt * dt

    # Return S, E, I, and R
    return S, E, I, R