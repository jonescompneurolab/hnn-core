:Comment : LVA ca channel. Note: mtau is an approximation from the plots
:Reference : :		original: Avery and Johnston 1996, tau from Randall 1997; adapted from Hay et al., 2011 PLOS Comp Bio, https://doi.org/10.1371/journal.pcbi.1002107

:Comment: shifted by -10 mv to correct for junction potential (in Hay et al, 2011)
:Comment: corrected rates using q10 = 2.3, target temperature 37 (adjusted by K.Duecker for duecker_ET_model), orginal 21

NEURON	{
	SUFFIX Ca_LVAst_hay2011
	USEION ca READ eca WRITE ica
	RANGE gbar, gCa_LVAst, ica
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
	eca	(mV)
	ica	(mA/cm2)
	gCa_LVAst	(S/cm2)
	mInf
	mTau    (ms)
	hInf
	hTau    (ms)
}

STATE	{
	m
	h
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	gCa_LVAst = gbar*m*m*h
	ica = gCa_LVAst*(v-eca)
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
  qt = 2.3^((37-21)/10)

	UNITSOFF
		v = v + 10
		mInf = 1.0000/(1+ exp((v - -30.000)/-6))
		mTau = (5.0000 + 20.0000/(1+exp((v - -25.000)/5)))/qt
		hInf = 1.0000/(1+ exp((v - -80.000)/6.4))
		hTau = (20.0000 + 50.0000/(1+exp((v - -40.000)/7)))/qt
		v = v - 10
	UNITSON
}
