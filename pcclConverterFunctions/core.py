import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as intrp
import scipy.misc as scpmsc
import warnings
"""

"""

# class interpolation():
#     """
#     The scipy interpolation callable objects work great but throw an 
#     error when you call them outside of the time range. Sometimes 
#     the scipy integrators need to make a call just slightly outside.
#     This is a wrapper function that performs a constant extrapolation
#     in the case that the argument is outside of the interpolation range
#     """
    
#     def __init__(self, time_list, data_list, kind="cubic"):
        
#         self.orig_interpolation = intrp.interp1d(time_list,
#                                                  data_list,
#                                                  kind=kind)

#         self.top_limit = time_list[-1]
#         self.bottom_limit = time_list[0]

#     def __call__(self, t):
        
#         try:
#             # first try to use the interpolation
#             result = self.orig_interpolation(t)
#             return result

#         # If it didnt work do the constant extrapolation
#         except ValueError as err:
#             emessage = err.args[0]
#             # if its above the interpolated range
#             if emessage == "A value in x_new is above the interpolation range.":

#                 warning_string = ("Value {} above the interpolation ".format(t) + 
#                                   "range limit of {}".format(self.top_limit))
#                 warnings.warn(warning_string)
                
#                 print(self.top_limit)
#                 result = self.orig_interpolation(self.top_limit)
#                 return result
                        

#             elif emessage == "A value in x_new is below the interpolation range.":
                
#                 warning_string = ("Value {} below the interpolation ".format(t) + 
#                                   "range limit of {}".format(self.bottom_limit))
#                 warnings.warn(warning_string)

#                 result = self.orig_interpolation(self.bottom_limit)
#                 return result
        
# Function to clean the "0.1" and spaces from the column names
def cleanColumns(name):
    if name[-1] == "1":
        name = name[0:-2]

    name = name.strip()
    return name

# Function to increase the resolution by linear interpolation
def increaseResolution(data):
    n = len(data)
    x = np.arange(0, 2*n, 2)
    x1 = np.arange(0,2*n -1 ,1)
    return  np.interp(x1, x, data)

def readCSV():

    pfile = 'PCCL_inputs/UtahHiawatha/primary.csv'
    sfile = 'PCCL_inputs/UtahHiawatha/secondary.csv'

    ## Read primary volatiles
    pccl_primary_df = pd.read_csv(pfile, header=1, sep=",",
                                  skipinitialspace=True)
    pccl_primary_df.rename(columns = cleanColumns, inplace=True)
    pccl_primary_df.rename(columns = {"time s":"time", "Tmp C":"T"}, 
                           inplace=True)

    ## Read secondary volatiles/tar products
    pccl_secondary_df = pd.read_csv(sfile, header=1, sep=",",
                                    skipinitialspace=True)
    pccl_secondary_df.rename(columns = cleanColumns, inplace=True)
    pccl_secondary_df.rename(columns = {"time s":"time", "Tmp C":"T"}, 
                             inplace=True)

    # Convert Temperature to Kelvin
    pccl_primary_df["T"] += 273.15
    pccl_secondary_df["T"] += 273.15

    # Add a nitrogen column, assume the fraction is some fraction of the 
    # particel daf initial mass
    pccl_primary_df = pccl_primary_df.assign(N2 = [50.0] * len(pccl_primary_df["T"]))
    pccl_secondary_df = pccl_secondary_df.assign(N2 = [50.0] * len(pccl_secondary_df["T"]))

    return ((pccl_primary_df, pccl_secondary_df))

def functionsFromTimeSeriesDf(df):
    """
    Takes the read in apandas df from csv and cubicly interpolates the 
    time series. Returns a dictionary. The dictionary 
    values are a time series function, the key is the column/species name.
    """

    # Initialize the dictionaries
    functions = {}
    
    # time will be needed for all series to interpolate against
    time = df["time"].tolist()

    # Calculate functions
    for column in df.columns.values:
        functions[column] = intrp.interp1d(time,
                                           df[column].tolist(),
                                           kind="cubic",
                                           bounds_error=False,
                                           fill_value="extrapolate")

    # Need time to be a list not a function
    functions["time"] = time

    return functions

def functionFromArray(time, data, new_time=None):
    """
    Takes an np array of data and time (same length). Performs a cubic interpolation
    and returns the callable interpolation object.
    
    If new_time is provided the function is first formed for the original `time' 
    then it a new interpolation object is formed with the new_time as a base. 
    This is useful if the original time array is too detailed.
    """

    data = np.array(data).flatten()
    time = np.array(time).flatten()

    function = intrp.interp1d(time,
                              data,
                              kind="cubic",
                              bounds_error=False,
                              fill_value="extrapolate")
    
    if new_time:
        new_time = np.array(time).flatten()
        
        function = intrp.interp1d(new_time,
                                  function(new_time),
                                  kind="cubic",
                                  bounds_error=False,
                                  fill_value="extrapolate")
        
    return function

def getDataFunctions():
    """ 
    Grabs data frames from csv for both prim and sec fields.
    Uses functionsFromTimeSeiesDf() fcn to find the 
    corresponding funcitons for each.
    """
    
    # Read CSVS
    (primary_df,secondary_df) = readCSV()

    # Add the secondary weight loss to secondary_df
    # it can be used later to determine soot yield 
    # when compared with the total primary weight loss.
    sd = secondary_df
    species_secondary = ['Tar', 'CO2', 'H2O', 'CO', 'HCN', 'CH4', 'C2H4', 
                      'C2H6', 'C3H6', 'C3H8', 'H2', 'H2S', 'Oils']

    sd["No Soot Weight Loss"] = pd.Series([0.0]* len(sd["Tar"]))
    for species in species_secondary:
        sd["No Soot Weight Loss"] += sd[species]

    primary_functions = functionsFromTimeSeriesDf(primary_df)
    secondary_functions = functionsFromTimeSeriesDf(secondary_df)

    # Wait until now to calculate soot because the primary and
    # secondary time series are different. Once they are functions
    # they are easier to work with.
    # There is no wt-loss in the secondary data.

    secondary_functions["Soot"] = lambda t: (primary_functions["Wt Loss"](t) - 
        secondary_functions["No Soot Weight Loss"](t))

    return (primary_functions, secondary_functions)


def calcTimeDerivative(function, times, dx=1e-6):
    """
    given function which is a python function
    that acts as a 1-d math function of time, calculate derivative.
    times is a list or numpy array of time that corresponds to the function.
    Central differnce is used and the spacing is specified by dx
    """

    # list of derivative value at each timestep
    deriv_list = [scpmsc.derivative(function,
                                    t,
                                    dx=dx)
                  for t in times[1:-1]]

    deriv_list = np.zeros(len(times))

    for (idx,time) in enumerate(times):
        try:
            deriv_list[idx] = scpmsc.derivative(function,
                                                time,
                                                dx=dx)

        # The function is an interpolation and cant be used 
        # for central differencing below or above the time interval
        # do either forward or backward differencing as appropriate
        except ValueError as err:
            below = 'A value in x_new is below the interpolation range.' 
            above = 'A value in x_new is above the interpolation range.'
            message = err.args[0]
            if message == below:
                # forward diff
                # should only happen at time[0]
                if not time == times[0]:
                    raise ValueError("Problem in forward difference")

                next_time = times[idx + 1]
                next_value = function(next_time)
                deriv_list[idx] = (next_value - function(time))/(next_time - time)

            if message == above:
                # Backward diff
                if not time == times[-1]:
                    raise ValueError("Problem in backward difference")

                prev_time = times[idx - 1]
                prev_value = function(prev_time)
                deriv_list[idx] = (function(time) - prev_value)/(time - prev_time)

    ## Now perform interpolation on the derivative list
    deriv_function = intrp.interp1d(times,
                                    deriv_list,
                                    kind="cubic",
                                    bounds_error=False,
                                    fill_value="extrapolate")
                
    return deriv_function


    
if __name__ == "__main__":
    (pf,sf) = getDataFunctions()

    deriv = calcTimeDerivative(pf["Tar"], pf['time'])

    ptime = pf["time"]
    stime = sf["time"]
    plt.plot(ptime, pf["Tar"](ptime)/max(pf["Tar"](ptime)), '.',
             ptime, deriv(ptime)/max(deriv(ptime)), '--')
    plt.show()

    
    
