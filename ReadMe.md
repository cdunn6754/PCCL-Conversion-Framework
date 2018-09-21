## PCCL Conversion Framework (PCF)

This is a pretty esoteric project. It relies on a commercial coal chemistry
analysis program named P.C. Coal Lab (PCCL). The high level picture is that
the PCF can take the output of this coal chemistry software and use its
predictions about devolatilization rates, devolatilization yields and coal
tar decomposition to fit rates which can be used in our CFD efforts. For
example the
[PCFSootTarFoam](https://github.com/cdunn6754/OpenFOAM_5.x_Applications/tree/master/lagrangian/PCFSootTarFoam)
solver uses the rates determined by the PCF to describe the how two competitive
tar consumption processes proceed, tar cracking and soot formation. For a
much more in-depth explanation of the theory behind this approach check
out my [thesis](https://clintdunn.org/static/thesis).

### Background:
PCCL performs a chemical analysis on coal particles base on their ultimate
proximate analyses. This analysis consists of predicting devolatilization
characteristics for a **single coal particle** in a
heating condition that is provided by the user.
One of the most prevalent volatile species is tar. Tar is not a particular
chemical species but instead refers to a group of large aromatic hydrocarbons.
In an inert environment (as considered here) these tars breakdown in two ways;
tar can decompose to form light gases and tar can be consumed to form soot.

PCCL can not only predict devolatilization but also fits rates for how these
tar consumption processes proceed. This information is very useful for our CFD
simulations in which we don't have the computational power to perform chemical
analyses for every single coal particle in the furnace; we just want the rates
which can be integrated affordably to predict these things.

The problem with this approach is that PCCL performs the analysis for a single
particle and formulates rate equations based on the particle mass fraction
yields.
These rate equations are not amenable to CFD simulations where the
rate equations are necessarily based on local gas phase mass fractions, i.e
mass fraction within a gaseous mixture, not the mass fraction yields from
PCCL (this distinction gets a little technical and confusing, please see
the
[thesis](https://clintdunn.org/static/thesis)
for more information).
The goal of the PCF then, is to use the information that PCCL provides about
devolatilization and tar consumption but to reformulate it to be used in
gas phase reaction rates that are commonly used in CFD modeling frameworks
(e.g. the Two Equation soot model from Brown 1996).

The overall workflow for the PCF
proceeds by first reading the PCCL output CSV files, that information
consists of time series, the time is that over which the coal particle is
heated and
decomposed, that describe mass yields fractions from the particle. This is the
raw information that is provided by PCCL that we are trying to use. There are
two sets of this info provided by PCCL, one called 'primary' and another called
'secondary' both provide time series mass yields. The difference is that the
primary set does not account for the breakdown of tar, once tar is yielded
it remains such that eventually the predicted tar yield is the same as the
ultimate yield ~ 30%. The secondary data set includes the prediction of tar
breakdown so at the end of the simulation, even though tar is produced all along,
it may all be consumed by soot formation and tar cracking.
The secondary data describes the environment as it actually is while the
primary data gives us a way to keep track of the rate of tar production from
devolatilization.

The next step is to convert both of these data sets from yield fractions to
more typical mass fractions, i.e. gaseous mass fractions of a mixture, that
can be used to inform rate equations in CFD.

The final step is to form a 0-D conservation equation for Tar that can be
used to optimize rates to fit the PCCL predictions that have been converted
to usable mass fractions. There are four free rate parameters to fit and
we just use some dumb brute force optimization tools from scipy.

### Content:

There are four main python files that drive everything.


* pcclConverterFunctions.core
	This module provides functions that are involved in reading the PCCL
	output csv files

* pcclConverterFunctions.chemistry
	This module is poorly named. I though at first that we would need to do some
	chemistry calculations to determine the density of the mixtures but that
	became unnecessary. Instead this module contains functions to convert those
	PCCL yield fractions into the mixture mass fractions that we need.

* comparison.py
  Contains a monolithic class, `SpeciesComparison`
	that tracks the time series of both PCCL secondary
	and primary information as well as the result of the integration
	of the CFD rates. It has functionality to integrate the CFD rates
	so that they can be compared to the PCCL predictions. This class
	holds the data that is then optimized over. It uses functions from both
	of the above modules.

* optimize.py
  Creates a `SpeciesComparison` instance that to hold the PCCL data and
	provide a way to integrate the rates. It then optimizes rate constants until
	the CFD model predictions match PCCL. There are two cost function functions
	that are used in the optimization and two 'optimizer' classes that are
	called by the actual scipy optimizer. The classes handle updating the
	`SpeciesComparison` instance when new rate constants are assigned.

### Using the PCF

The first step is to obtain and appropriately modify PCCL output files. There
are some examples given in `PCCL_inputs/`. Unfortunately the path to the
csv files needs to be hardcoded into the core.py module. You can then just
run optimizer.py and the optimization will begin. Unfortunately much of the
flow of the optimization is still hard-coded right now and I am going to leave
it that way. I also have some plotting afterwards that produced figures for my
thesis.
