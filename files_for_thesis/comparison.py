import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as intgrt
from timeit import default_timer as timer

import  pcclConverterFunctions.core as cr
import  pcclConverterFunctions.chemistry as ch

"""
dY_ts/dt = S_primary - S_sootform - S_cracking

 S_primary - The source of primary tar, i.e. the rate of primary tar creation dY_tp/dt
 S_sf - Soot formation rate.
 S_cr - Cracking or gasification of tar, rate.
"""

class SpeciesComparison:
    """
    Class to hold a comparison between PCCL and TT model.

    TODO: Maybe we could make these objects represent a particular coal
    and pass the name in automate the csv reading.
    """

    def __init__(self):

        # get the data function dictionaries
        (pf,sf) = cr.getDataFunctions()

        # Temperature functions
        self.pTemp = pf["T"]
        self.sTemp = sf["T"]
        # Get the time lists, the only list elements of the pf/sf dictionaries 
        self.ptime = pf["time"]
        self.stime = sf["time"]

        # Primary and secondary tar are central to many calculations
        self.pc_ptar_mf_function = ch.getTarMassFractionFunctions(pf,self.ptime)
        self.pc_star_mf_function = ch.getTarMassFractionFunctions(sf,self.stime)

        # Also need the entire secondary speices list
        self.s_all_mf_functions = ch.getMassFractionFunctions(sf,self.stime)
        self.p_all_mf_functions = ch.getMassFractionFunctions(pf,self.ptime)


        # The rate at which primary tar mass fraction grows, this is the 
        # source for the secondary tar
        self.p_source_function = cr.calcTimeDerivative(self.pc_ptar_mf_function, 
                                                       self.ptime)

        # TT model default rate constants from Brown 1998
        # In order (A_sf, E_sf, A_cr, E_cr)
        self.rate_constants = (5.02e8, 198.9, 9.77e10, 286.9)

        # Default time array over which to integrate the 
        # soot and cracking rates to find the tt model mass fractions
        self.int_time = np.arange(0.0, self.stime[-1], 1e-6)

        # The tt model integrated secondary tar list.
        self.tt_star_mf_list = intgrt.odeint(self.star_rate, 
                                             [self.pc_star_mf_function(0.0)], 
                                             self.stime)

        # The interpolated function version of tt_star_mf_list.
        # These are pretty useful when you can't carry around the corresponding
        # time arrary for a list.
        ## TODO make a properties setup so that this is 
        # automatically calculated when the list is updated/modified.
        self.tt_star_mf_function = cr.functionFromArray(np.array(self.stime), 
                                                        self.tt_star_mf_list)


    def star_rate(self, star_mf, t):
        """
        Form star derivative function to hand to scipy integrator for
        dY_st/dt, the secondary tar overall rate. This is the 
        derivative at a particular time t.
        """

        T = self.sTemp(t)
        Sp = self.p_source_function(t)
        return ch.formSecondaryTarRate(star_mf,
                                       Sp,
                                       T,
                                       self.rate_constants)

    def soot_rate(self, _, t):
        """
        The mass fraction of soot produced from tar breakdown.
        This is the rate at a particular time t which specifies
        the secondary tar mass fraction and temperature.
        """
        T = self.sTemp(t)
        star_mf = self.tt_star_mf_function(t)
        # soot formation rate contants
        A = self.rate_constants[0]
        E = self.rate_constants[1]
        return ch.formRateAtTemp(star_mf, T, A, E)


    def gas_rate(self, _, t):
        """
        The mass fraction of gas that is a product of tar decomposition
        but isnt soot.
        This is the rate at a particular time t which specifies
        the secondary tar mass fraction and temperature.
        """
        T = self.sTemp(t)
        star_mf = self.tt_star_mf_function(t)
        # cracking rate constants
        A = self.rate_constants[2]
        E = self.rate_constants[3]
        return ch.formRateAtTemp(star_mf, T, A, E)

    def integrateRates(self,
                       rate_constants, 
                       dt = 1e-6):
        """
        Runs the TT model to predict the secondary tar, soot and 
        tar cracking gas product mass fractions time series. Whici can then 
        be compared with the PCCL results.
        
        rate_constants should be a 4-tuple of (A_sf, E_sf, A_cr, E_cr).
        This will be called during optimization to reintegrate the 
        rate equations with these new constants.

        dt is the time step over which to integrate. 
        The default, set in the function above, seems to work fine.
        
        Returns: a 4-tuple of arrays (star_mf, soot_mf, gas_mf, int_time)
         that represent time series
          star_mf - Mass fraction of secondary tar
          soot_mf -       ""         soot
          gas_mf -        ""         gas
          int_time - The time array that corresponds to these mass fraction arrays.   
        """

        self.int_time = np.arange(0.0, self.stime[-1], dt)

        self.rate_constants = rate_constants

        # integrate to find the trends in time.
        self.tt_star_mf_list = intgrt.odeint(self.star_rate, 
                                             [self.pc_star_mf_function(0.0)], 
                                             self.int_time)

        # Need a callable for the calculation of soot and cracking rates
        self.tt_star_mf_function = cr.functionFromArray(self.int_time,
                                                        self.tt_star_mf_list,
                                                        self.stime)


        # The I.C.s will be zero anyway but eventually we can 
        #TODO replace the hard-coded 0.0 with soot_mf_function(0.0) ...
        soot_mf_list = intgrt.odeint(self.soot_rate, [0.0], self.int_time)
        gas_mf_list = intgrt.odeint(self.gas_rate, [0.0], self.int_time)

        return (self.tt_star_mf_list, soot_mf_list, gas_mf_list, self.int_time)
        

if __name__ == "__main__":


    # Starting point for the constants
    A_sf = 2.33e7#5.02e8
    E_sf = 154.7#198.9
    A_cr = 9.77e10
    E_cr = 286.9

    rate_constants = (A_sf, E_sf, A_cr, E_cr)
    cracking_rate_constants = rate_constants[2:]
    soot_rate_constants = rate_constants[0:2]

    c = SpeciesComparison()

    
    ## PCCL arrays
    stime = c.stime
    pccl_soot_mf = c.s_all_mf_functions["Soot"](stime)
    pccl_star_mf = c.pc_star_mf_function(stime)
    pccl_ptar_mf = c.pc_ptar_mf_function(stime)
    pccl_svol_mf = pccl_ptar_mf - pccl_star_mf - pccl_soot_mf

    ## TT model arrays
    # results : (star, soot, svol, int_time)
    tt_mfs = c.integrateRates(rate_constants)
    int_time = tt_mfs[3]
    

    ## PCCL plots
    plt.figure(2)
    plt.plot(stime, pccl_star_mf,'--', label="PCCL secondary")
    plt.plot(stime, pccl_ptar_mf,'-', label="PCCL primary")
    plt.plot(stime, pccl_soot_mf, '-.', label="PCCL Soot")
    plt.plot(stime, pccl_svol_mf, ':', label="PCCL secondary Volatiles")
    plt.plot(int_time, tt_mfs[0], '--', label="TT secondary tar")
    plt.plot(int_time, tt_mfs[1], '-.', label="TT soot")
    plt.plot(int_time, tt_mfs[2], ':', label="TT secondary volatiles")
    plt.title("Unoptimized TT rates, taken directly from Brown 1998")
    plt.legend()  
    
    ##
    plt.figure(3)
    plt.plot(stime, pccl_star_mf,'--', label="PCCL secondary tar")
    plt.plot(stime, pccl_ptar_mf,'-', label="PCCL primary tar")
    plt.plot(stime, pccl_soot_mf, '-.', label="PCCL soot")
   # plt.plot(stime, pccl_svol_mf, ':', label="PCCL secondary Volatiles")
    plt.xlabel("Time [s]")
    plt.ylabel("Mass Fraction")
    plt.legend()    
    

    plt.show()
