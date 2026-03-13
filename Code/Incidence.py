import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


# Day numbers
t = np.array([
1,2,3,4,5,6,7,8,9,10,
11,12,13,14,15,16,17,18,19,20,
21,22,23,24,25,26,27,28,29,30,
31,32,33,34,35,36,37,38,39,40,
41,42,43,44,45
])



# Active reported daily cases
N = np.array([
1,1,1,1,1,1,2,2,2,3,
3,4,4,4,5,6,7,9,9,10,
13,14,16,17,20,24,25,31,33,38,
43,54,56,60,75,76,93,94,110,134,
155,170,189,211,223
])

# Log-transform the cumulative cases
logN = np.log(N)

# Fit a straight line to log(N) vs time
slope, intercept, r_value, p_value, std_err = linregress(t, logN)

lambda_growth = slope
N0 = np.exp(intercept)

print("Growth rate λ:", lambda_growth)
print("Initial cases N0:", N0)

# Compute instantaneous incidence at each time point
incidence = lambda_growth * N

print("Incidence (new cases per day):")
print(incidence)


