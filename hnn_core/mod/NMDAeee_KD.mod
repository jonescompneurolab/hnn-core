TITLE simple NMDA receptors

COMMENT
-----------------------------------------------------------------------------

Essentially the same as /examples/nrniv/netcon/ampa.mod in the NEURON
distribution - i.e. Alain Destexhe's simple AMPA model - but with
different binding and unbinding rates and with a magnesium block.
Modified by Andrew Davison, The Babraham Institute, May 2000


	Simple model for glutamate AMPA receptors
	=========================================

  - FIRST-ORDER KINETICS, FIT TO WHOLE-CELL RECORDINGS

    Whole-cell recorded postsynaptic currents mediated by AMPA/Kainate
    receptors (Xiang et al., J. Neurophysiol. 71: 2552-2556, 1994) were used
    to estimate the parameters of the present model; the fit was performed
    using a simplex algorithm (see Destexhe et al., J. Computational Neurosci.
    1: 195-230, 1994).

  - SHORT PULSES OF TRANSMITTER (0.3 ms, 0.5 mM)

    The simplified model was obtained from a detailed synaptic model that
    included the release of transmitter in adjacent terminals, its lateral
    diffusion and uptake, and its binding on postsynaptic receptors (Destexhe
    and Sejnowski, 1995).  Short pulses of transmitter with first-order
    kinetics were found to be the best fast alternative to represent the more
    detailed models.

  - ANALYTIC EXPRESSION

    The first-order model can be solved analytically, leading to a very fast
    mechanism for simulating synapses, since no differential equation must be
    solved (see references below).



References

   Destexhe, A., Mainen, Z.F. and Sejnowski, T.J.  An efficient method for
   computing synaptic conductances based on a kinetic model of receptor binding
   Neural Computation 6: 10-14, 1994.

   Destexhe, A., Mainen, Z.F. and Sejnowski, T.J. Synthesis of models for
   excitable membranes, synaptic transmission and neuromodulation using a
   common kinetic formalism, Journal of Computational Neuroscience 1:
   195-230, 1994.

Orignal file by:
Kiki Sidiropoulou
Adjusted Cdur = 1 and Beta= 0.01 for better nmda spikes
PROCEDURE rate: FROM -140 TO 80 WITH 1000

Modified by:
  Joe Graham joe.w.graham@gmail.com
  2017/09/18

From:
  https://senselab.med.yale.edu/ModelDB/showmodel.cshtml?model=230811&file=/PapoutsiEtAl2017/mod_files/NMDA.mod#tabs-2

* changed calcium handling
* changed Beta from 0.01 to 0.0015 to get nice plateaus


Further modified by:
  Katharina Duecker
  
  Changed such that calcium would decay back to baseline within 450 ms (see Gordon, Polsky, Schiller, 2006, J Neuro

-----------------------------------------------------------------------------
ENDCOMMENT



NEURON {
	POINT_PROCESS NMDAeee_KD
	RANGE g, Alpha, Beta, e, ica, Cdur, gmax, i
	USEION ca READ cai,cao WRITE ica :USEION ca WRITE ica
	NONSPECIFIC_CURRENT  iNMDA
	GLOBAL mg, Cmax, fracca
}
UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(umho) = (micromho)
	(mM) = (milli/liter)
    FARADAY = (faraday) (coulomb)
    R = (k-mole) (joule/degC)
}

PARAMETER {
	Cmax	= 1	     (mM)       : max transmitter concentration
	Cdur	= 1	     (ms)		: transmitter duration (rising phase)
	Alpha	= 3      (/ms /mM)	: forward (binding) rate (4)
	Beta 	= 0.003 (/ms) :0.0015 (/ms)      : reverse (unbinding) rate
	e	    = 0	     (mV)		: reversal potential - Try -10 mV on 04/03/2018, but no good results
    mg      = 1      (mM)       : external magnesium concentration
    fracca  = 0.13   : fraction of current that is ca ions; Spruston et al 95
	gmax  = 1      (uS)
}


ASSIGNED {
	v		(mV)		: postsynaptic voltage
	iNMDA 	(nA)    	: current = g*(v - e)
	i (nA)
	g 		(umho)		: conductance
	Rinf				: steady state channels open
	Rtau	(ms)	    : time constant of channel binding
	synon
    B                   : magnesium block
	ica
    cai     (mM)
    cao     (mM)
}

STATE {Ron Roff}

INITIAL {
	Rinf = Cmax*Alpha / (Cmax*Alpha + Beta)
	Rtau = 1 / (Cmax*Alpha + Beta)
	synon = 0
}

BREAKPOINT {
	SOLVE release METHOD cnexp
    	B = mgblock(v)
	g = (Ron + Roff) * B * gmax
	iNMDA = g * (v - e) * (1-fracca) : iNMDA = g*(v - e)
    if(fracca > 0.0){ica = g * ghkg(v,cai,cao,1.5) * fracca}
	i = iNMDA + ica
}

INCLUDE "ghk.inc"

DERIVATIVE release {
	Ron' = (synon * Rinf - Ron) / Rtau
	Roff' = -Beta * Roff
}

FUNCTION mgblock(v(mV)) {
        TABLE
        DEPEND mg
        FROM -140 TO 80 WITH 1000

	 mgblock = 1 / (1 + exp(0.062 (/mV) * -v) * (mg / 3.57 (mM)))
     : from Jahr & Stevens, 1990 J Neurosci
}

: following supports both saturation from single input and
: summation from multiple inputs
: if spike occurs during CDur then new off time is t + CDur
: ie. transmitter concatenates but does not summate
: Note: automatic initialization of all reference args to 0 except first

NET_RECEIVE(weight, on, nspike, r0, t0 (ms)) {
	: flag is an implicit argument of NET_RECEIVE and normally 0
        if (flag == 0) { : a spike, so turn on if not already in a Cdur pulse
		nspike = nspike + 1
		if (!on) {
			r0 = r0 * exp(-Beta * (t - t0))
			t0 = t
			on = 1
			synon = synon + weight
			state_discontinuity(Ron, Ron + r0)
			state_discontinuity(Roff, Roff - r0)
		}
:		 come again in Cdur with flag = current value of nspike
		net_send(Cdur, nspike)
       }
	if (flag == nspike) { : if this associated with last spike then turn off
		r0 = weight * Rinf + (r0 - weight * Rinf) * exp(-(t - t0) / Rtau)
		t0 = t
		synon = synon - weight
		state_discontinuity(Ron, Ron - r0)
		state_discontinuity(Roff, Roff + r0)
		on = 0
	}
}
