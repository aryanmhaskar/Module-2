#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Load the data
data = pd.read_csv('../Data/mystery_virus_daily_active_counts_RELEASE#1.csv', parse_dates=['date'], header=0, index_col=None)

#%%
# Make a plot of the active cases over time

#%% import libraries
import numpy as np
import matplotlib.pyplot as plt



#%% define infection data

# Day numbers
days = np.array([
1,2,3,4,5,6,7,8,9,10,
11,12,13,14,15,16,17,18,19,20,
21,22,23,24,25,26,27,28,29,30,
31,32,33,34,35,36,37,38,39,40,
41,42,43,44,45
])



# Active reported daily cases
active_cases = np.array([
1,1,1,1,1,1,2,2,2,3,
3,4,4,4,5,6,7,9,9,10,
13,14,16,17,20,24,25,31,33,38,
43,54,56,60,75,76,93,94,110,134,
155,170,189,211,223
])



#%% plot active infections vs day

fig, ax = plt.subplots(figsize=(10, 6))

plt.plot(days, active_cases, marker='o', color='red', label='Active Infections')

plt.title('DATA RELEASE #1: Day vs Active Infections')
plt.xlabel('Day')
plt.ylabel('Active Reported Daily Cases')
plt.legend()
plt.grid(True)

plt.show()



'''

What do you notice about the initial infections?

Initially, the infection spreads slowly. The graph shows 1 active case per day for the
first 6 days. This increases to 2 cases on day 7, and then 3 cases on day 10. 



How could we measure how quickly its spreading?

The infection is spreading exponentially, so we could measure the infection growth rate
with the equation I(t) = I_0 * e^(rt), where I(t) is the number of infections at time t,
I-0 is the initial number of infections, r is the growth rate, and t is time.



What information about the virus would be helpful in determining the shape of the
outbreak curve?

It would be helpful to know the incubation period of the virus because if someone is
infected, they may not show symptoms for several days. A delay in symptom onset can
affect the timing of when cases are reported and can influence the shape of the outbreak
curve. The infectious period would also be helpful to know because it would indicate the
period of time during which the virus can be transmitted to others, so if the infectious
period is long, it could lead to more cases and a prolonged outbreak curve. 

'''
