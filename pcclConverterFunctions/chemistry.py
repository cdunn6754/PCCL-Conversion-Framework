import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as intrp
import cantera as ct



"""
This module uses the time series from core to determine the 
mass fractions of each species as a time series.
For now the species considered, whether they are primary or secondary, 
are hard-coded. 
"""

species_considered = ['Tar', 'CO2', 'H2O', 'CO', 'HCN', 'CH4', 'C2H4', 'Soot',
                      'C2H6', 'C3H6', 'C3H8', 'H2', 'H2S', 'Oils', 'N2']

def getTarMassFractionFunctions(functions, times):
    """
     The functions are yield daf we need to convert to 
    find the tar mass fraction of each species in the enviroment
    surrounding the particle. To do this add up all daf yield 
    fractions of  volatile species including Tar from the
    secondary information and then renormalize the Tar fraction
    """

    # list of sums to be used for renormalization
    sum_list = np.zeros(len(times))

    name_set = set(species_considered).intersection(functions.keys())

    for name in name_set:
        for (time_idx,time) in enumerate(times):
            sum_list[time_idx] += np.round(functions[name](time),8)
            
    # the list of tar daf yields to be renormalized
    tar_yield_list = np.round(functions["Tar"](times),8)
    
    tar_mf_list = tar_yield_list/(sum_list + 1e-15)
    
    tar_mf_function = intrp.interp1d(times,
                                     tar_mf_list,
                                     kind="cubic")

    return tar_mf_function


def getMassFractionFunctions(functions, times):
    """
    The same as getTarMassFractions but it returns a dictionary
    with the mass fraction time series for all of the relevant species
    """

    # list of sums to be used for renormalization
    sum_list = np.zeros(len(times))

    name_set = set(species_considered).intersection(functions.keys())

    for name in name_set:
        for (time_idx,time) in enumerate(times):
            sum_list[time_idx] += np.round(functions[name](time),8)

    # intialize the dict that is eventually returned
    mf_dict = {}
        
    # loop through species and calculate the actual mass fractions
    # by renormalizing the yield fractions
    for name in name_set:
        
        # the list of name species daf yields to be renormalized by sum_list
        yield_list = np.round(functions[name](times),8)
    
        # renormalize by sum_list
        mf_list = yield_list/(sum_list + 1e-15)
    
        # then reinterpolate list to form a function
        mf_dict[name] = intrp.interp1d(times, mf_list, kind="cubic")

    return mf_dict

def getMassFractionDictAtTime(mfs, time):
    """
    Given the dictionary of mass fraction functions, mfs, this returns a 
    dictionary of mass fractions with names as keys 
    at a particular time, time.

    """
    mf_dict = {}
    
    for name in mfs.keys():
        mf_dict[name] = mfs[name](time)

    return mf_dict

def getDensity(mfs, temperature, times):
    """
    given a dictionary of interpolated mass fraction functions, mfs, and their
    corresponding time arrays , times, this function returns an
    interpolated function for the density along the same times array

    Some of the PCCL stuff isnt defined in the cantera mech we use
    so we are forced to rename/remove some things
      H2S  - Moved into H2

    It is also important to be careful because the first few times steps
    often have no species at all. Obviously Cantera will have a difficult
    time intuiting a density out of the ether. To prevent this we first
    check if there is any tar mass fraction. If that mass fraction is zero
    then we set the density to -1.0. The idea is that when calculating rates
    if Y_tar is zero then the density wont matter. I hope that if this function
    is ever repurposed it should indicate there is a problem when a negative
    density arises.
    """

    # needed properties
    temperature_list = temperature(times)
    pressure = ct.one_atm
    
    density_list = np.zeros(len(times))
    
    
    # create Cantera gas mixture with huge mech
    gas = ct.Solution('cantera_mechanisms/gri_PCCLConvert.cti')

    for (time_idx,time) in enumerate(times):

        time_mfs = getMassFractionDictAtTime(mfs, time)

        # Here we do the replacement to work with the mechanism
        time_mfs["H2"] += time_mfs["H2S"]
        del time_mfs["H2S"]

        if time_mfs["Tar"] > 0.0:
            # Set the mixture at intial contents
            gas.TPY = temperature_list[time_idx], pressure, time_mfs
            density_list[time_idx] = gas.density
        else:
            density_list[time_idx] = -1.0

    density_function = intrp.interp1d(times, density_list, kind="cubic")
    
    return density_function

def formRateFunction(tar_sec, stime, stemp, A, E):
    """
    Given the secondary tar mass fraction function, secondary time function,
    secondary temperature function, density function and 
    the rate constants calculate the
    rate in units of [1/s] i.e. Y/time, for either soot formation or
    tar cracking. Depends on the rate constants given, the formulas 
    are otherwise identical

    r = Y_st * A * exp(-E/RT) / rho
    """
    R = 8.315e-3 #kJ/[mol K]
        
    tar_list = tar_sec(stime)
    temp_list = stemp(stime)

    rate_list = np.round(tar_list * A * np.exp(-E / (R * temp_list)),15)

    return intrp.interp1d(stime, rate_list, kind="cubic")

def formRateAtTime(tar_sec, temp, A, E):
    """
    Instead of calculating the rates from the pccl known secondary tar
    mass fraction for the entire time series as in formTarFunction(), here
    we just give scalar values for the current secondary tar mass fraction and
    temperature. Along with the rate constants we use that info to calculate
    the rates and then return it, again a single scalar value.
    """
    R = 8.315e-3 #kJ/[mol K]
 
    return tar_sec * A * np.exp(-E / (R * temp))

def formSecondaryTarRate(star_mf, primSource, T):
    """
    Form the dY_st/dt rate from the primary tar source,
    with soot formation and tar cracking sink terms.
    dY_st/dt = dY_p/dt - r_sf - r_cr
    
    - star_mf is the scalar secondary tar mass fraction
    - prim_source is the current value of dY_p/dt
    - T is the current temperature
    """

    # Rate constants from Josehpson 2016/Brown 1998
    A_sf = 5.02e8
    E_sf = 198.9
    A_cr = 9.77e10
    E_cr = 286.9

    # soot formation rate
    r_sf = formRateAtTime(star_mf, T, A_sf, E_sf)
    # cracking
    r_cr = formRateAtTime(star_mf, T, A_cr, E_cr)

    return primSource - r_sf - r_cr
