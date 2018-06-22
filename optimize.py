import matplotlib.pyplot as plt

import comparison as cpn


c = cpn.SpeciesComparison()




# Starting point for the constants
A_sf = 5.02e8
E_sf = 198.9
A_cr = 9.77e10
E_cr = 286.9

rate_constants = (A_sf, E_sf, A_cr, E_cr)

results = c.integrateRates(rate_constants)

int_time = results[3]

plt.plot(int_time, results[0], '-', label="TT Secondary Tar")
plt.plot(int_time, results[1], '-.', label="TT Soot")
plt.plot(int_time, results[2], ':', label="TT Gas")
plt.plot(c.stime, c.star_mf_function(c.stime), '--', label="PCCL Secondary Tar")
plt.plot(c.stime, c.s_all_mf_functions["Soot"](c.stime), '-', label="PCCL Soot")
plt.legend()
plt.show()
