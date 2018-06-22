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
        self.ptar_mf_function = ch.getTarMassFractionFunctions(pf,self.ptime)
        self.star_mf_function = ch.getTarMassFractionFunctions(sf,self.stime)

        # Also need the entire secondary speices list
        self.s_all_mf_functions = ch.getMassFractionFunctions(sf,self.stime)

        # The rate at which primary tar mass fraction grows, this is the 
        # source for the secondary tar
        self.p_source_function = cr.calcTimeDerivative(self.ptar_mf_function, 
                                                       self.ptime)

        # TT model default rate constants from Brown 1998
        # In order (A_sf, E_sf, A_cr, E_cr)
        self.rate_constants = (5.02e8, 198.9, 9.77e10, 286.9)

    ## Form derivative function to hand to scipy integrator for
    #  dY_st/dt, the secondary tar overall rate.
    def star_rate(self, star_mf, t):

        T = self.sTemp(t)
        Sp = self.p_source_function(t)
        return ch.formSecondaryTarRate(star_mf,
                                       Sp,
                                       T,
                                       self.rate_constants)

    # The mass fraction of soot produced from tar breakdown
    def soot_rate(self, _, t):
        T = self.sTemp(t)
        star_mf = self.star_mf_function(t)
        A = self.rate_constants[0]
        E = self.rate_constants[1]
        return ch.formRateAtTemp(star_mf, T, A, E)

    # The mass fraction of gas that is a product of tar decomposition
    # but isnt soot
    def gas_rate(self, _, t):
        T = self.sTemp(t)
        star_mf = self.star_mf_function(t)
        A = self.rate_constants[2]
        E = self.rate_constants[3]
        return ch.formRateAtTemp(star_mf, T, A, E)

    def integrateRates(self,
                       rate_constants, 
                       dt = 1e-6):
        """
        Runs the TT model to predict the secondary tar, soot and 
        tar cracking gas product mass fractions. Can then be compared
        with the PCCL results.
        
        rate_constants should be a 4-tuple of (A_sf, E_sf, A_cr, E_cr)
        This will be called during optimization to reintegrate the 
        rate equations with these new constants.

        int_time is the array of times over which to integrate. 
        The default, set in the function seems to work fine.
        
        Returns: a 4-tuple of arrays (star_mf, soot_mf, gas_mf, int_time)
          star_mf - Mass fraction of secondary tar
          soot_mf -       ""         soot
          gas_mf -        ""         gas
          int_time - The time array that corresponds to these mass fraction arrays.   
        """

        int_time = np.arange(0.0, self.stime[-1], dt)

        self.rate_constants = rate_constants

        # integrate to find the trends in time.
        star_mf_list = intgrt.odeint(self.star_rate, 
                                     [self.star_mf_function(0.0)], 
                                     int_time)
        # The I.C.s will be zero but eventually we can 
        #TODO replace the hard-coded 0.0 with soot_mf_function(0.0) ...
        soot_mf_list = intgrt.odeint(self.soot_rate, [0.0], int_time)
        gas_mf_list = intgrt.odeint(self.gas_rate, [0.0], int_time)

        return (star_mf_list, soot_mf_list, gas_mf_list, int_time)
        
        

        
# def compare():
#     # get the data function dictionaries
#     (pf,sf) = cr.getDataFunctions()

#     # get the temperature functions
#     pTemp = pf["T"]
#     sTemp = sf["T"]

    
#     # Get the time lists, the only list elements of the pf/sf dictionaries 
#     ptime = pf["time"]
#     stime = sf["time"]

#     # get the interpolated time series function of tar mf
#     # that is within the enviroment surrounding the particle
#     # will be used to form cracking and soot formation rates from Brown
#     star_mf_function = ch.getTarMassFractionFunctions(sf,stime)
    
#     # We need the primary fraction too, not daf yielded but
#     # fraction around particle. This way we use all primary data
#     # so the Tar has not decomposed. I think that is right as 
#     # this is being used to form the primary tar source term.
#     ptar_mf_function = ch.getTarMassFractionFunctions(pf,ptime)

#     ## Now we are in a position to integrate the dY_t/dt equation

#     # First get the primary source derivative interpolated function
#     # This is the derivative of the primary tar mass fraction in time.
#     pSource_function = cr.calcTimeDerivative(ptar_mf_function, ptime)

#     # Mass fraction functions in time for all secondary species
#     s_all_mf_functions = ch.getMassFractionFunctions(sf,stime)

#     # Rate constants from Josehpson 2016/Brown 1998
#     A_sf = 5.02e8
#     E_sf = 198.9
#     A_cr = 9.77e10
#     E_cr = 286.9

#     # integration time list, we will use a little higher definition here than PCCL
#     int_time = np.arange(1e-6, stime[-1], 1e-6)

#     ## Form derivative function to hand to scipy integrator for
#     #  dY_st/dt, the secondary tar overall rate.
#     def star_rate(star_mf, t):
#         T = sTemp(t)
#         Sp = pSource_function(t)
#         return ch.formSecondaryTarRate(star_mf, Sp, T)

#     # Same for soot formation
#     # dY_s/dt
#     def soot_rate(_, t):
#         T = sTemp(t)
#         star_mf = star_mf_function(t)
#         return ch.formRateAtTemp(star_mf, T, A_sf, E_sf)

#     # Same for tar cracking gas formation
#     # dY_g/dt
#     def gas_rate(_, t):
#         T = sTemp(t)
#         star_mf = star_mf_function(t)
#         return ch.formRateAtTemp(star_mf, T, A_cr, E_cr)

#     # Integrate with scipy
#     star_mf = intgrt.odeint(star_rate, [star_mf_function(0.0)], int_time)
#     soot_mf = intgrt.odeint(soot_rate, [0.0], int_time)
#     gas_mf = intgrt.odeint(gas_rate, [0.0], int_time)

#     return (star_mf, soot_mf, gas_mf)


# if __name__ == "__main__":
#     # plots to compare PCCL and TT model outputs
#     plt.figure(3)
#     plt.plot(int_time, star_mf, '-', label="TT secondary")
#     plt.plot(int_time, soot_mf, '-.', label="TT soot")
#     plt.plot(int_time, gas_mf, '-.', label="TT gas")
#     plt.plot(stime, star_mf_function(stime),'--', label="PCCL secondary")
#     plt.plot(stime, ptar_mf_function(stime),'--', label="PCCL primary")
#     plt.plot(stime, s_all_mf_functions["Soot"](stime), ':', label="PCCL Soot")
#     plt.legend()


#     plt.show()
