import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as intgrt

import  pcclConverterFunctions.core as cr
import  pcclConverterFunctions.chemistry as ch

"""
dY_ts/dt = S_primary - S_sootform - S_cracking

 S_primary - The source of primary tar, i.e. the rate of primary tar creation dY_tp/dt
 S_sf - Soot formation rate.
 S_cr - Cracking or gasification of tar, rate.

TOD: should make a comparison class that could be instantiated for a particular 
species. It would hold the PCCL results and perform the TT model integration too.
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
star_mf_function = ch.getTarMassFractionFunctions(sf,stime)

# We need the primary fraction too, not daf yielded but
# fraction around particle. This way we use all primary data
# so the Tar has not decomposed. I think that is right as 
# this is being used to form the primary tar source term.
ptar_mf_function = ch.getTarMassFractionFunctions(pf,ptime)

## Now we are in a position to integrate the dY_t/dt equation

# First get the primary source derivative interpolated function
# This is the derivative of the primary tar mass fraction in time.
pSource_function = cr.calcTimeDerivative(ptar_mf_function, ptime)

# Calculate the soot formation and cracking rates in units [1/s] (Y_t/s)
s_all_mf_functions = ch.getMassFractionFunctions(sf,stime)

# Rate constants from Josehpson 2016/Brown 1998
A_sf = 5.02e8
E_sf = 198.9
A_cr = 9.77e10
E_cr = 286.9

# integration time list, we will use a little higher definition here than PCCL
int_time = np.arange(1e-6, stime[-1], 1e-6)

## Form derivative function to hand to scipy integrator for
#  dY_st/dt, the secondary tar overall rate.
def star_rate(star_mf, t):
    T = sTemp(t)
    Sp = pSource_function(t)
    return ch.formSecondaryTarRate(star_mf, Sp, T)

# Same for soot formation
# dY_s/dt
def soot_rate(_, t):
    T = sTemp(t)
    star_mf = star_mf_function(t)
    return ch.formRateAtTemp(star_mf, T, A_sf, E_sf)

# Same for tar cracking gas formation
# dY_g/dt
def gas_rate(_, t):
    T = sTemp(t)
    star_mf = star_mf_function(t)
    return ch.formRateAtTemp(star_mf, T, A_cr, E_cr)

# Integrate with scipy
star_mf = intgrt.odeint(star_rate, [star_mf_function(0.0)], int_time)
soot_mf = intgrt.odeint(soot_rate, [0.0], int_time)
gas_mf = intgrt.odeint(gas_rate, [0.0], int_time)

# plots to compare PCCL and TT model outputs
plt.figure(3)
plt.plot(int_time, star_mf, '-', label="TT secondary")
plt.plot(int_time, soot_mf, '-.', label="TT soot")
plt.plot(int_time, gas_mf, '-.', label="TT gas")
plt.plot(stime, star_mf_function(stime),'--', label="PCCL secondary")
plt.plot(stime, ptar_mf_function(stime),'--', label="PCCL primary")
plt.plot(stime, s_all_mf_functions["Soot"](stime), ':', label="PCCL Soot")
plt.legend()


plt.show()
exit()


# class comparisonSpeciesBase:
#     """
#     Base class to hold a species comparison between PCCL and TT model.
#     """

#     def __init__(self, species_name, sf_rate_constants, cr_rate_constants):
#         self.name = species_name

#         self.Asf = sf_rate_constants[0]
#         self.Esf = sf_rate_constants[1]

#         self.Acr = cr_rate_constants[0]
#         self.Ecr = cr_rate_constants[1]

#         # get the data function dictionaries
#         (pf,sf) = cr.getDataFunctions()
