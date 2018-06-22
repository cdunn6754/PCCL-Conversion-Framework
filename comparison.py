import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as intgrt

import  pcclConverterFunctions.core as cr
import  pcclConverterFunctions.chemistry as ch

"""
dY_ts/dt = S_p - S_sootform - S_cracking

 S_p - The source of primary tar, i.e. the rate of primary tar creation dY_tp/dt
 S_sf - Soot formation rate

"""

# get the data function dictionaries
(pf,sf) = cr.getDataFunctions()

# get the temperature functions
pTemp = pf["T"]
sTemp = sf["T"]


# Get the time lists, the only list elements of the pf/sf dictionaries 
ptime = pf["time"]
stime = sf["time"]

# get the interpolated time series function of tar mf
# that is within the enviroment surrounding the particle
# will be used to form cracking and soot formation rates from Brown
tar_sec_mf = ch.getTarMassFractionFunctions(sf,stime)

# We need the primary fraction too, not daf yielded but
# fraction around particle. This way we use all primary data
# so the Tar has not decomposed. I think that is right as 
# this is being used to form the source term S_p
tar_prim_mf = ch.getTarMassFractionFunctions(pf,ptime)

## Now we are in a position to integrate the dY_t/dt equation

#First get the primary source derivative interpolated function
S_p = cr.calcTimeDerivative(tar_prim_mf, ptime)



# Calculate the soot formation and cracking rates in units [1/s] (Y_t/s)
mass_fractions_all = ch.getMassFractionFunctions(sf,stime)
#density_function = ch.getDensity(mass_fractions_all, sTemp, stime)

# Rate constants from Josehpson 2016/Brown 1998
A_sf = 5.02e8
E_sf = 198.9
A_cr = 9.77e10
E_cr = 286.9

# rate_sf = ch.formRateFunction(tar_sec_mf, stime, 
#                               sTemp,
#                               A_sf, E_sf)

# rate_cr = ch.formRateFunction(tar_sec_mf, stime, 
#                               sTemp,
#                               A_cr, E_cr)


# plt.figure(1)
# plt.plot(ptime, S_p(ptime), "-", label="Primary")
# plt.plot(stime, rate_sf(stime), '.', label="Soot Formation")        
# plt.plot(stime, rate_cr(stime), '--', label="Cracking")
# plt.legend()

# plt.plot(ptime, tar_prim_mf(ptime)/max(tar_prim_mf(ptime)), '-',
#          ptime, S_p(ptime)/max(S_p(ptime)), '--')


# Now we can form the total tar rate dY_t/dt function
# def tar_rate(y,t): 
#     return S_p(t) - rate_sf(t) - rate_cr(t)

#result = intgrt.quad(tar_rate, 0.0, 0.0211)


## Now do it for real, in the previous integration we used the
# predetermined PCCL secondary tar mass fraction to calculate the rates.
# In reality we dont have that information and are integrating the equation to 
# determine it. So now we will do that. We integrate the rates and then
# need to recalculate the mass fraction of secondary tar each time.


# integration time list, we will use a little higher definition here than PCCL
int_time = np.arange(1e-6, stime[-1], 1e-6)

# initial secondary tar mass fraction
star_mf = np.zeros(len(int_time))
star_mf[0] = tar_sec_mf(0.0)

# Soot formed from tar decomposition
soot_mf = np.zeros(len(int_time))

# Other volatiles/gases from tar cracking/gasification
gas_mf = np.zeros(len(int_time))


# Secondary tar formation at time 0.0
total_rate = ch.formSecondaryTarRate(star_mf[0], S_p(0.0), sTemp(0.0))
soot_rate = ch.formRateAtTemp(star_mf[0], sTemp(0.0), A_sf, E_sf)
gas_rate = ch.formRateAtTemp(star_mf[0], sTemp(0.0), A_cr, E_cr)

for (time_idx,time) in enumerate(int_time[:-1]):
    """
    iterate through time steps excluding the first.
    Use a forward rule to 
    integrate as phi(t + 1) = phi(t) + dt * phi'(t)
    - total_rate is phi'(t-1)
    - curr_rate is phi'(t)
    """
    dt = int_time[time_idx + 1] - time
    
    star_mf[time_idx + 1] = star_mf[time_idx] + dt * total_rate
    soot_mf[time_idx + 1] = soot_mf[time_idx] + dt * soot_rate
    gas_mf[time_idx + 1] = gas_mf[time_idx] + dt * gas_rate

    total_rate = ch.formSecondaryTarRate(star_mf[time_idx + 1], 
                                         S_p(time + dt), 
                                         sTemp(time + dt))
    soot_rate = ch.formRateAtTemp(star_mf[time_idx + 1],
                                  sTemp(time + dt),
                                  A_sf, E_sf)
    gas_rate = ch.formRateAtTemp(star_mf[time_idx + 1],
                                  sTemp(time + dt),
                                  A_cr, E_cr)



plt.figure(2)
plt.plot(int_time, star_mf, '-', label="TT secondary")
plt.plot(int_time, soot_mf, '-.', label="TT soot")
plt.plot(int_time, gas_mf, '-.', label="TT gas")
plt.plot(stime, tar_sec_mf(stime),'--', label="PCCL secondary")
plt.plot(stime, tar_prim_mf(stime),'--', label="PCCL primary")
plt.plot(stime, mass_fractions_all["Soot"](stime), ':', label="PCCL Soot")
plt.legend()
plt.show()
    

exit()


