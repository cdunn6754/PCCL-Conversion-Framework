import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
    

usecols = np.arange(0,16,1) + 35
usecols = [0] + [3] + list(usecols)
pccl_df = pd.read_csv('PCCoResults.csv', header=4, sep=",", 
                      skipinitialspace=True, usecols=usecols)

# Rename them
pccl_df.rename(columns = cleanColumns, inplace=True)
pccl_df.rename(columns = {"time s":"time", "Stime s":"Stime", "Tmp C":"T"}, 
               inplace=True)

# print(pccl_df.columns.values)
# exit()

## Get the columns we need
Y_p = pccl_df["Tar"].tolist()
t = pccl_df["time"].tolist()
t_s = pccl_df["Stime"].tolist()
T = pccl_df["T"].tolist()
Y_s = pccl_df["STar"].tolist()

# get rid of Nan in the primary fields
Y_p = [x for x in Y_p if not str(x) == "nan"]
t = [x for x in t if not str(x) == "nan"]
# generate some finer resolution
Y_p = increaseResolution(Y_p)
t = increaseResolution(t)



## Central difference to find primary tar derivative

# derivative d Y_p/dt = S_p
S_p = np.zeros(len(Y_p))

# do a forward diff on the first
S_p[0] = (S_p[1] - S_p[0])/ (t[1] - t[0])
# and a backward onthe last
S_p[-1] = (S_p[-1] - S_p[-2])/ (t[-1] - t[-2])

# for the rest do a central diff
for (idx, _) in enumerate(Y_p):
    # skip the first and last
    if idx == 0 or idx == len(Y_p) -1:
        continue
    
    t1 = t[idx - 1]
    t2 = t[idx + 1]
    y1 = Y_p[idx - 1]
    y2 = Y_p[idx + 1]

    S_p[idx] = (y2 - y1)/(t2-t1)


## Now form the derivatives S_soot and S_crack
        
    
