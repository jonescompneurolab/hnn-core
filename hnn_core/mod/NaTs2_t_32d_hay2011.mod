:Reference :Colbert and Pan 2002, from Hay et al., 2011 PLOS Comp Bio, https://doi.org/10.1371/journal.pcbi.1002107
:comment: took the NaTa and shifted both activation/inactivation by 6 mv
:Comment : corrected rates using q10 = 2.3, target temperature 32 (K. Duecker for duecker_ET_model, slower L2/3 pyr APs), orginal 21

NEURON	{
	SUFFIX NaTs2_t_32d_hay2011
	USEION na READ ena WRITE ina
	RANGE gbar, gNaTs2_t, ina
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gbar = 0.00001 (S/cm2)
}

ASSIGNED	{
	v	(mV)
	ena	(mV)
	ina	(mA/cm2)
	gNaTs2_t	(S/cm2)
	mInf
	mTau    (ms)
	mAlpha
	mBeta
	hInf
	hTau    (ms)
	hAlpha
	hBeta
}

STATE	{
	m
	h
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	gNaTs2_t = gbar*m*m*m*h
	ina = gNaTs2_t*(v-ena)
}

DERIVATIVE states	{
	rates()
	m' = (mInf-m)/mTau
	h' = (hInf-h)/hTau
}

INITIAL{
	rates()
	m = mInf
	h = hInf
}

PROCEDURE rates(){
  LOCAL qt
  qt = 2.3^((32-21)/10)

	UNITSOFF
    if(v == -32){
    	v = v+0.0001
    }
		mAlpha = (0.182 * (v- -32))/(1-(exp(-(v- -32)/6)))
		mBeta  = (0.124 * (-v -32))/(1-(exp(-(-v -32)/6)))
		mInf = mAlpha/(mAlpha + mBeta)
		mTau = (1/(mAlpha + mBeta))/qt

    if(v == -60){
      v = v + 0.0001
    }
		hAlpha = (-0.015 * (v- -60))/(1-(exp((v- -60)/6)))
		hBeta  = (-0.015 * (-v -60))/(1-(exp((-v -60)/6)))
		hInf = hAlpha/(hAlpha + hBeta)
		hTau = (1/(hAlpha + hBeta))/qt
	UNITSON
}
