: Dynamics that track inside calcium concentration
: modified from Destexhe et al. 1994

NEURON	{
	SUFFIX CaDynamics_E2
	USEION ca READ ica WRITE cai
	RANGE decay, gamma, minCai, depth
}

UNITS	{
	(mV) = (millivolt)
	(mA) = (milliamp)
	FARADAY = (faraday) (coulombs)
	(molar) = (1/liter)
	(mM) = (millimolar)
	(um)	= (micron)
}

PARAMETER	{
	gamma = 0.05 : percent of free calcium (not buffered)
	decay = 80 (ms) : rate of removal of calcium
	depth = 0.1 (um) : depth of shell
	minCai = 5e-5 (mM)
}

ASSIGNED	{ica (mA/cm2)
		drive_channel (mM/ms)}

STATE	{
	cai (mM)
	}

BREAKPOINT	{ SOLVE states METHOD cnexp }

DERIVATIVE states	{
	: KD: added to avoid Ca2+ leaving through NMDA when cell depolarized
	drive_channel = -(10000)*(ica*gamma/(2*FARADAY*depth))
	if (drive_channel <= 0.) { drive_channel = 0.  }
	cai' = drive_channel - (cai - minCai)/decay
}
