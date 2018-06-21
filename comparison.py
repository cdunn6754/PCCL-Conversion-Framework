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
density_function = ch.getDensity(mass_fractions_all, sTemp, stime)

# Rate constants from Josehpson 2016/Brown 1998
A_sf = 5.02e8
E_sf = 198.9
A_cr = 9.77e10
E_cr = 286.9

rate_sf = ch.formRateFunction(tar_sec_mf, stime, 
                              sTemp, density_function,
                              A_sf, E_sf)

rate_cr = ch.formRateFunction(tar_sec_mf, stime, 
                              sTemp, density_function,
                              A_cr, E_cr)

plt.plot(ptime, S_p(ptime), "-", stime, rate_sf(stime), '.',
         stime, rate_cr(stime), '--')

# plt.plot(ptime, tar_prim_mf(ptime)/max(tar_prim_mf(ptime)), '-',
#          ptime, S_p(ptime)/max(S_p(ptime)), '--')
plt.show()

# Now we can form the total tar rate dY_t/dt function
tar_rate = lambda t: S_p(t) - rate_sf(t) - rate_cr(t)

result = intgrt.quad(tar_rate, 0.0, 0.0211)

print(result[0])


exit()


