05-29-18:
	The goal of this project is to compare the secondary tar concentrations	
from PCCL to a 0-D reaction with the fletcher model. We form a diff eq like

     	  dY_s/dt = dY_p/dt - S_soot - S_cracking

Which neglects combustion. The S_i terms come from the fletcher model and we will stick
with the PCCL primary devol rate for tar as dY_p/dt. Integrating this equation should
hopefully yield a Y_s trend that is similar to the one predicted in PCCL.
