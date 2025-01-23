COMMENT
Conceptual model:  D current for a model of a fast-spiking cortical interneuron.

Authors and citation:
  Golomb D, Donner K, Shacham L, Shlosberg D, Amitai Y, Hansel D (2007).
  Mechanisms of Firing Patterns in Fast-Spiking Cortical Interneurons. 
  PLoS Comput Biol 3:e156.

Original implementation and programming language/simulation environment:
  by Golomb et al. for XPP
  Available from http://senselab.med.yale.edu/modeldb/ShowModel.asp?model=97747

This implementation is by N.T. Carnevale and V. Yamini for NEURON.
ENDCOMMENT

NEURON {
  SUFFIX kd
  USEION k READ ek WRITE ik
  RANGE gbar, g
}

UNITS {
  (S) = (siemens)
  (mV) = (millivolt)
  (mA) = (milliamp)
}

PARAMETER {
  gbar = 0.00039 (S/cm2)
  theta_a = -50 (mV)
  sigma_a = 20 (mV)
  theta_b = -70 (mV)
  sigma_b = -6 (mV)
  tau_a = 2 (ms)
  tau_b = 150 (ms)
}

ASSIGNED {
  v (mV)
  ek (mV)
  ik (mA/cm2)
  g (S/cm2)
}

STATE {a b}

BREAKPOINT {
  SOLVE states METHOD cnexp
  g = gbar * a^3 * b
  ik = g * (v-ek)
}

INITIAL {
  a = ainfi(v)
  b = binfi(v)
}

DERIVATIVE states {
  a' = (ainfi(v)-a)/tau_a
  b' = (binfi(v)-b)/tau_b
}

FUNCTION ainfi(v (mV)) {
  UNITSOFF
  ainfi=1/(1 + exp(-(v-theta_a)/sigma_a))
  UNITSON
}

FUNCTION binfi(v (mV)) {
  UNITSOFF
  binfi=1/(1 + exp(-(v-theta_b)/sigma_b))
  UNITSON
}
