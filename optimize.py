import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import pickle

import comparison as cpn


def getSumSquares(x_1, x_2):
    """
    Takes two arrays, returns the sum(sqr(x_1[i] - x_2[i]))
    """
    return np.sum([ (x_1_i - x_2_i)**2 for (x_1_i,x_2_i) in zip(x_1,x_2)])

def integrateDiff(x_1, x_2):
    """
    Same as getSumSquares put just sums the abs of the difference
    """
    return np.sum([ abs(x_1_i - x_2_i) for (x_1_i,x_2_i) in zip(x_1,x_2)])

class SootObjective:

    def __init__(self, cracking_rate_constants, c):
        self.c = c
        self.cracking_rate_constants = cracking_rate_constants


    def __call__(self, soot_rate_constants):
        rate_constants_ = (soot_rate_constants[0],
                            soot_rate_constants[1],
                            self.cracking_rate_constants[0],
                            self.cracking_rate_constants[1])

        #Integrate the TT model
        int_results = self.c.integrateRates(rate_constants_)
        int_time = int_results[3]

        pccl_soot_list = list(self.c.s_all_mf_functions["Soot"](int_time))
        tt_soot_list = list(int_results[1])

        cost = integrateDiff(pccl_soot_list, tt_soot_list)

        print("New constants: {}, {}".format(soot_rate_constants[0],
                                            soot_rate_constants[1]))

        print("New integral value: {}\n".format(cost))

        return cost

class StarObjective:
    def __init__(self, soot_rate_constants, c):
        self.c = c
        self.soot_rate_constants = soot_rate_constants


    def __call__(self, cracking_rate_constants):
        rate_constants_ = (self.soot_rate_constants[0],
                            self.soot_rate_constants[1],
                            cracking_rate_constants[0],
                            cracking_rate_constants[1])

        #Integrate the TT model
        int_results = self.c.integrateRates(rate_constants_)
        int_time = int_results[3]

        pccl_star_list = list(self.c.pc_star_mf_function(int_time))
        tt_star_list = list(int_results[0])

        #cost = integrateDiff(pccl_star_list, tt_star_list)
        cost = integrateDiff(pccl_star_list, tt_star_list)

        print("New constants: {}, {}".format(cracking_rate_constants[0],
                                            cracking_rate_constants[1]))

        print("New integral value: {}\n".format(cost))

        return cost


# Starting point for the constants
A_sf = 5.02e8
E_sf = 198.9
A_cr = 9.77e10
E_cr = 286.9

rate_constants = (A_sf, E_sf, A_cr, E_cr)
cracking_rate_constants = rate_constants[2:]
soot_rate_constants = rate_constants[0:2]

c = cpn.SpeciesComparison()


# soot_obj = SootObjective(cracking_rate_constants, c)
# min_result = opt.minimize(soot_obj, soot_rate_constants, method='Nelder-Mead',
#                           options={'fatol':1e-3})
# exit()


# New soot constants
# rate_constants = (2.33e7, 154.7, A_cr, E_cr)
# star_obj = StarObjective(rate_constants, c)
# min_result = opt.minimize(star_obj, soot_rate_constants, method='Nelder-Mead',
#                           options={'fatol':1e-3})
# exit()

rate_constants = (2.33e7, 154.7, 1988.4, 47.31)
results = c.integrateRates(rate_constants)

int_time = results[3]


pccl_svol_mf = (c.pc_ptar_mf_function(c.stime) - 
                c.pc_star_mf_function(c.stime) - 
                c.s_all_mf_functions["Soot"](c.stime))

plt.plot(int_time, results[0], '--', label="TT Secondary Tar")
plt.plot(int_time, results[1], '-.', label="TT Soot")
plt.plot(int_time, results[2], ':', label="TT Gas")
plt.plot(c.stime, c.pc_ptar_mf_function(c.stime), '-', label="PCCL Primary Tar")
plt.plot(c.stime, c.pc_star_mf_function(c.stime), '--', label="PCCL Secondary Tar")
plt.plot(c.stime, c.s_all_mf_functions["Soot"](c.stime), '-.', label="PCCL Soot")
plt.plot(c.stime, pccl_svol_mf , ':', label="PCCL secondary volatile Gas")
plt.legend()
plt.title("Rates optimized to fit PCCL predictions")



plt.figure(5)
temps = np.arange(273,1900, 0.1)
rate = lambda T,A,E: A * np.exp(-E/(0.008314 * T))
plt.plot(temps, rate(temps,rate_constants[0], rate_constants[1]), label="soot")
plt.plot(temps, rate(temps,rate_constants[2],rate_constants[3]), label="cracking")
plt.legend()
plt.show()


