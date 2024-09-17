/* Created by Language version: 7.7.0 */
/* NOT VECTORIZED */
#define NRN_VECTORIZED 0
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mech_api.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__km
#define _nrn_initial _nrn_initial__km
#define nrn_cur _nrn_cur__km
#define _nrn_current _nrn_current__km
#define nrn_jacob _nrn_jacob__km
#define nrn_state _nrn_state__km
#define _net_receive _net_receive__km 
#define _f_trates _f_trates__km 
#define rates rates__km 
#define states states__km 
#define trates trates__km 
 
#define _threadargscomma_ /**/
#define _threadargsprotocomma_ /**/
#define _threadargs_ /**/
#define _threadargsproto_ /**/
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 static double *_p; static Datum *_ppvar;
 
#define t nrn_threads->_t
#define dt nrn_threads->_dt
#define gbar _p[0]
#define gbar_columnindex 0
#define gk _p[1]
#define gk_columnindex 1
#define ninf _p[2]
#define ninf_columnindex 2
#define ntau _p[3]
#define ntau_columnindex 3
#define n _p[4]
#define n_columnindex 4
#define a _p[5]
#define a_columnindex 5
#define b _p[6]
#define b_columnindex 6
#define ik _p[7]
#define ik_columnindex 7
#define ek _p[8]
#define ek_columnindex 8
#define Dn _p[9]
#define Dn_columnindex 9
#define _g _p[10]
#define _g_columnindex 10
#define _ion_ek	*_ppvar[0]._pval
#define _ion_ik	*_ppvar[1]._pval
#define _ion_dikdv	*_ppvar[2]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 /* external NEURON variables */
 extern double celsius;
 /* declaration of user functions */
 static void _hoc_rates(void);
 static void _hoc_trates(void);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _p = _prop->param; _ppvar = _prop->dparam;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_km", _hoc_setdata,
 "rates_km", _hoc_rates,
 "trates_km", _hoc_trates,
 0, 0
};
 /* declare global and static user variables */
#define Rb Rb_km
 double Rb = 0.001;
#define Ra Ra_km
 double Ra = 0.001;
#define qa qa_km
 double qa = 9;
#define q10 q10_km
 double q10 = 2.3;
#define tha tha_km
 double tha = -30;
#define tshift tshift_km
 double tshift = 30.7;
#define tadj tadj_km
 double tadj = 0;
#define temp temp_km
 double temp = 23;
#define usetable usetable_km
 double usetable = 1;
#define vmax vmax_km
 double vmax = 100;
#define vmin vmin_km
 double vmin = -120;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "usetable_km", 0, 1,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "tha_km", "mV",
 "qa_km", "mV",
 "Ra_km", "/ms",
 "Rb_km", "/ms",
 "temp_km", "degC",
 "vmin_km", "mV",
 "vmax_km", "mV",
 "gbar_km", "pS/um2",
 "gk_km", "pS/um2",
 "ntau_km", "ms",
 0,0
};
 static double delta_t = 1;
 static double n0 = 0;
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "tha_km", &tha_km,
 "qa_km", &qa_km,
 "Ra_km", &Ra_km,
 "Rb_km", &Rb_km,
 "temp_km", &temp_km,
 "q10_km", &q10_km,
 "tshift_km", &tshift_km,
 "vmin_km", &vmin_km,
 "vmax_km", &vmax_km,
 "tadj_km", &tadj_km,
 "usetable_km", &usetable_km,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(NrnThread*, _Memb_list*, int);
static void nrn_state(NrnThread*, _Memb_list*, int);
 static void nrn_cur(NrnThread*, _Memb_list*, int);
static void  nrn_jacob(NrnThread*, _Memb_list*, int);
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(NrnThread*, _Memb_list*, int);
static void _ode_matsol(NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[3]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"km",
 "gbar_km",
 0,
 "gk_km",
 "ninf_km",
 "ntau_km",
 0,
 "n_km",
 0,
 0};
 static Symbol* _k_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 11, _prop);
 	/*initialize range parameters*/
 	gbar = 10;
 	_prop->param = _p;
 	_prop->param_size = 11;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_k_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0]._pval = &prop_ion->param[0]; /* ek */
 	_ppvar[1]._pval = &prop_ion->param[3]; /* ik */
 	_ppvar[2]._pval = &prop_ion->param[4]; /* _ion_dikdv */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _km_reg() {
	int _vectorized = 0;
  _initlists();
 	ion_reg("k", -10000.);
 	_k_sym = hoc_lookup("k_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 0);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 11, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 km /Users/katharinaduecker/Documents/Projects Brown/hnn-core/hnn_core/mod/km.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double _znexp ;
 static double *_t_ninf;
 static double *_t_ntau;
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int _f_trates(double);
static int rates(double);
static int trates(double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static void _n_trates(double);
 static int _slist1[1], _dlist1[1];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 () {_reset=0;
 {
   trates ( _threadargscomma_ v ) ;
   Dn = ( ninf - n ) / ntau ;
   }
 return _reset;
}
 static int _ode_matsol1 () {
 trates ( _threadargscomma_ v ) ;
 Dn = Dn  / (1. - dt*( ( ( ( - 1.0 ) ) ) / ntau )) ;
  return 0;
}
 /*END CVODE*/
 static int states () {_reset=0;
 {
   trates ( _threadargscomma_ v ) ;
    n = n + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / ntau)))*(- ( ( ( ninf ) ) / ntau ) / ( ( ( ( - 1.0 ) ) ) / ntau ) - n) ;
   }
  return 0;
}
 static double _mfac_trates, _tmin_trates;
 static void _check_trates();
 static void _check_trates() {
  static int _maktable=1; int _i, _j, _ix = 0;
  double _xi, _tmax;
  static double _sav_celsius;
  static double _sav_temp;
  static double _sav_Ra;
  static double _sav_Rb;
  static double _sav_tha;
  static double _sav_qa;
  if (!usetable) {return;}
  if (_sav_celsius != celsius) { _maktable = 1;}
  if (_sav_temp != temp) { _maktable = 1;}
  if (_sav_Ra != Ra) { _maktable = 1;}
  if (_sav_Rb != Rb) { _maktable = 1;}
  if (_sav_tha != tha) { _maktable = 1;}
  if (_sav_qa != qa) { _maktable = 1;}
  if (_maktable) { double _x, _dx; _maktable=0;
   _tmin_trates =  vmin ;
   _tmax =  vmax ;
   _dx = (_tmax - _tmin_trates)/199.; _mfac_trates = 1./_dx;
   for (_i=0, _x=_tmin_trates; _i < 200; _x += _dx, _i++) {
    _f_trates(_x);
    _t_ninf[_i] = ninf;
    _t_ntau[_i] = ntau;
   }
   _sav_celsius = celsius;
   _sav_temp = temp;
   _sav_Ra = Ra;
   _sav_Rb = Rb;
   _sav_tha = tha;
   _sav_qa = qa;
  }
 }

 static int trates(double _lv){ _check_trates();
 _n_trates(_lv);
 return 0;
 }

 static void _n_trates(double _lv){ int _i, _j;
 double _xi, _theta;
 if (!usetable) {
 _f_trates(_lv); return; 
}
 _xi = _mfac_trates * (_lv - _tmin_trates);
 if (isnan(_xi)) {
  ninf = _xi;
  ntau = _xi;
  return;
 }
 if (_xi <= 0.) {
 ninf = _t_ninf[0];
 ntau = _t_ntau[0];
 return; }
 if (_xi >= 199.) {
 ninf = _t_ninf[199];
 ntau = _t_ntau[199];
 return; }
 _i = (int) _xi;
 _theta = _xi - (double)_i;
 ninf = _t_ninf[_i] + _theta*(_t_ninf[_i+1] - _t_ninf[_i]);
 ntau = _t_ntau[_i] + _theta*(_t_ntau[_i+1] - _t_ntau[_i]);
 }

 
static int  _f_trates (  double _lv ) {
   rates ( _threadargscomma_ _lv ) ;
    return 0; }
 
static void _hoc_trates(void) {
  double _r;
    _r = 1.;
 trates (  *getarg(1) );
 hoc_retpushx(_r);
}
 
static int  rates (  double _lv ) {
   a = Ra * ( _lv - tha ) / ( 1.0 - exp ( - ( _lv - tha ) / qa ) ) ;
   b = - Rb * ( _lv - tha ) / ( 1.0 - exp ( ( _lv - tha ) / qa ) ) ;
   tadj = pow( q10 , ( ( celsius - temp - tshift ) / 10.0 ) ) ;
   ntau = 1.0 / tadj / ( a + b ) ;
   ninf = a / ( a + b ) ;
    return 0; }
 
static void _hoc_rates(void) {
  double _r;
   _r = 1.;
 rates (  *getarg(1) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ return 1;}
 
static void _ode_spec(NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ek = _ion_ek;
     _ode_spec1 ();
  }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 1; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 ();
 }
 
static void _ode_matsol(NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ek = _ion_ek;
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_k_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_k_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_k_sym, _ppvar, 2, 4);
 }

static void initmodel() {
  int _i; double _save;_ninits++;
 _save = t;
 t = 0.0;
{
  n = n0;
 {
   trates ( _threadargscomma_ v ) ;
   n = ninf ;
   }
  _sav_indep = t; t = _save;

}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
  ek = _ion_ek;
 initmodel();
 }}

static double _nrn_current(double _v){double _current=0.;v=_v;{ {
   gk = tadj * gbar * n ;
   ik = ( 1e-4 ) * gk * ( v - ek ) ;
   }
 _current += ik;

} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
  ek = _ion_ek;
 _g = _nrn_current(_v + .001);
 	{ double _dik;
  _dik = ik;
 _rhs = _nrn_current(_v);
  _ion_dikdv += (_dik - ik)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ik += ik ;
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}}

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
  ek = _ion_ek;
 { error =  states();
 if(error){fprintf(stderr,"at line 90 in file km.mod:\n    SOLVE states METHOD cnexp\n"); nrn_complain(_p); abort_run(error);}
 } }}

}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = n_columnindex;  _dlist1[0] = Dn_columnindex;
   _t_ninf = makevector(200*sizeof(double));
   _t_ntau = makevector(200*sizeof(double));
_first = 0;
}

#if NMODL_TEXT
static const char* nmodl_filename = "/Users/katharinaduecker/Documents/Projects Brown/hnn-core/hnn_core/mod/km.mod";
static const char* nmodl_file_text = 
  "COMMENT\n"
  "    26 Ago 2002 Modification of original channel to allow variable time step\n"
  "            and to correct an initialization error.\n"
  "\n"
  "    Done by Michael Hines (michael.hines@yale.edu) and Ruggero Scorcioni (rscorcio@gmu.edu)\n"
  "            at EU Advance Course in Computational Neuroscience. Obidos, Portugal\n"
  "\n"
  "    km.mod\n"
  "\n"
  "    Potassium channel, Hodgkin-Huxley style kinetics\n"
  "    Based on I-M (muscarinic K channel)\n"
  "    Slow, noninactivating\n"
  "\n"
  "    Original Author: Zach Mainen, Salk Institute, 1995, zach@salk.edu\n"
  "ENDCOMMENT\n"
  "\n"
  "INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}\n"
  "\n"
  "NEURON {\n"
  "    SUFFIX km\n"
  "    USEION k READ ek WRITE ik\n"
  "    RANGE n, gk, gbar\n"
  "    RANGE ninf, ntau\n"
  "    GLOBAL Ra, Rb\n"
  "    GLOBAL q10, temp, tadj, vmin, vmax, tshift\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "    (mA) = (milliamp)\n"
  "    (mV) = (millivolt)\n"
  "    (pS) = (picosiemens)\n"
  "    (um) = (micron)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "    : 0.03 mho/cm2\n"
  "    gbar = 10   (pS/um2)\n"
  "    v           (mV)\n"
  "\n"
  "    : v 1/2 for inf\n"
  "    tha = -30   (mV)\n"
  "\n"
  "    : inf slope\n"
  "    qa = 9      (mV)\n"
  "\n"
  "    : max act rate  (slow)\n"
  "    Ra = 0.001  (/ms)\n"
  "\n"
  "    : max deact rate  (slow)\n"
  "    Rb = 0.001  (/ms)\n"
  "\n"
  "    dt          (ms)\n"
  "    celsius     (degC)\n"
  "\n"
  "    : original temp\n"
  "    temp = 23   (degC)\n"
  "\n"
  "    : temp sensitivity\n"
  "    q10  = 2.3\n"
  "\n"
  "    tshift = 30.7\n"
  "\n"
  "    vmin = -120 (mV)\n"
  "    vmax = 100  (mV)\n"
  "}\n"
  "\n"
  "\n"
  "ASSIGNED {\n"
  "    a       (/ms)\n"
  "    b       (/ms)\n"
  "    ik      (mA/cm2)\n"
  "    gk      (pS/um2)\n"
  "    ek      (mV)\n"
  "    ninf\n"
  "    ntau    (ms)\n"
  "    tadj\n"
  "}\n"
  "\n"
  "\n"
  "STATE {\n"
  "    n\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "    trates(v)\n"
  "    n = ninf\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "    SOLVE states METHOD cnexp\n"
  "    gk = tadj * gbar * n\n"
  "    ik = (1e-4) * gk * (v - ek)\n"
  "}\n"
  "\n"
  "LOCAL nexp\n"
  "\n"
  ": Computes state variable n at the current v and dt.\n"
  "DERIVATIVE states {\n"
  "    trates(v)\n"
  "    n' = (ninf - n) / ntau\n"
  "}\n"
  "\n"
  ": Computes rate and other constants at current v.\n"
  ": Call once from HOC to initialize inf at resting v.\n"
  "PROCEDURE trates(v) {\n"
  "    TABLE ninf, ntau\n"
  "    DEPEND  celsius, temp, Ra, Rb, tha, qa\n"
  "\n"
  "    FROM vmin TO vmax WITH 199\n"
  "\n"
  "    : not consistently executed from here if usetable_hh == 1\n"
  "    rates(v)\n"
  "    : tinc = -dt * tadj\n"
  "    : nexp = 1 - exp(tinc/ntau)\n"
  "}\n"
  "\n"
  ": Computes rate and other constants at current v.\n"
  ": Call once from HOC to initialize inf at resting v.\n"
  "PROCEDURE rates(v) {\n"
  "    a = Ra * (v - tha) / (1 - exp(-(v - tha) / qa))\n"
  "    b = -Rb * (v - tha) / (1 - exp((v - tha) / qa))\n"
  "\n"
  "    tadj = q10^((celsius - temp - tshift) / 10)\n"
  "    ntau = 1/tadj/(a+b)\n"
  "    ninf = a/(a+b)\n"
  "}\n"
  ;
#endif
