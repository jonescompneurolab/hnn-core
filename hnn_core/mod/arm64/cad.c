/* Created by Language version: 7.7.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
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
 
#define nrn_init _nrn_init__cad
#define _nrn_initial _nrn_initial__cad
#define nrn_cur _nrn_cur__cad
#define _nrn_current _nrn_current__cad
#define nrn_jacob _nrn_jacob__cad
#define nrn_state _nrn_state__cad
#define _net_receive _net_receive__cad 
#define state state__cad 
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargsprotocomma_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
#define _threadargsproto_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define taur _p[0]
#define taur_columnindex 0
#define ca _p[1]
#define ca_columnindex 1
#define cai _p[2]
#define cai_columnindex 2
#define Dca _p[3]
#define Dca_columnindex 3
#define ica _p[4]
#define ica_columnindex 4
#define drive_channel _p[5]
#define drive_channel_columnindex 5
#define v _p[6]
#define v_columnindex 6
#define _g _p[7]
#define _g_columnindex 7
#define _ion_ica	*_ppvar[0]._pval
#define _ion_cai	*_ppvar[1]._pval
#define _style_ca	*((int*)_ppvar[2]._pvoid)
 
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
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 /* declaration of user functions */
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
 _extcall_prop = _prop;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_cad", _hoc_setdata,
 0, 0
};
 /* declare global and static user variables */
#define cainf cainf_cad
 double cainf = 0.0001;
#define depth depth_cad
 double depth = 0.1;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "depth_cad", "um",
 "cainf_cad", "mM",
 "taur_cad", "ms",
 "ca_cad", "mM",
 0,0
};
 static double ca0 = 0;
 static double delta_t = 1;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "depth_cad", &depth_cad,
 "cainf_cad", &cainf_cad,
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
 static void _ode_synonym(int, double**, Datum**);
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"cad",
 "taur_cad",
 0,
 0,
 "ca_cad",
 0,
 0};
 static Symbol* _ca_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 8, _prop);
 	/*initialize range parameters*/
 	taur = 200;
 	_prop->param = _p;
 	_prop->param_size = 8;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_ca_sym);
 nrn_check_conc_write(_prop, prop_ion, 1);
 nrn_promote(prop_ion, 3, 0);
 	_ppvar[0]._pval = &prop_ion->param[3]; /* ica */
 	_ppvar[1]._pval = &prop_ion->param[1]; /* cai */
 	_ppvar[2]._pvoid = (void*)(&(prop_ion->dparam[0]._i)); /* iontype for ca */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 "ca_cad", 1e-05,
 0,0
};
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _cad_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("ca", -10000.);
 	_ca_sym = hoc_lookup("ca_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 8, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "#ca_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "cvodeieq");
 	nrn_writes_conc(_mechtype, 0);
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_synonym(_mechtype, _ode_synonym);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 cad /Users/katharinaduecker/Documents/Projects Brown/hnn-core/hnn_core/mod/cad.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 
#define FARADAY _nrnunit_FARADAY[_nrnunit_use_legacy_]
static double _nrnunit_FARADAY[2] = {0x1.78e555060882cp+16, 96485.3}; /* 96485.3321233100141 */
static int _reset;
static char *modelname = "Decay of internal calcium concentration";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[1], _dlist1[1];
 static int state(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   drive_channel = - ( 10000.0 ) * ica / ( 2.0 * FARADAY * depth ) ;
   if ( drive_channel <= 0. ) {
     drive_channel = 0. ;
     }
   Dca = drive_channel + ( cainf - ca ) / taur ;
   cai = ca ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 drive_channel = - ( 10000.0 ) * ica / ( 2.0 * FARADAY * depth ) ;
 if ( drive_channel <= 0. ) {
   drive_channel = 0. ;
   }
 Dca = Dca  / (1. - dt*( ( ( ( - 1.0 ) ) ) / taur )) ;
 cai = ca ;
  return 0;
}
 /*END CVODE*/
 static int state (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
   drive_channel = - ( 10000.0 ) * ica / ( 2.0 * FARADAY * depth ) ;
   if ( drive_channel <= 0. ) {
     drive_channel = 0. ;
     }
    ca = ca + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / taur)))*(- ( drive_channel + ( ( cainf ) ) / taur ) / ( ( ( ( - 1.0 ) ) ) / taur ) - ca) ;
   cai = ca ;
   }
  return 0;
}
 
static int _ode_count(int _type){ return 1;}
 
static void _ode_spec(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ica = _ion_ica;
  cai = _ion_cai;
  cai = _ion_cai;
     _ode_spec1 (_p, _ppvar, _thread, _nt);
  _ion_cai = cai;
 }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 1; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 static void _ode_synonym(int _cnt, double** _pp, Datum** _ppd) { 
	double* _p; Datum* _ppvar;
 	int _i; 
	for (_i=0; _i < _cnt; ++_i) {_p = _pp[_i]; _ppvar = _ppd[_i];
 _ion_cai =  ca ;
 }}
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 (_p, _ppvar, _thread, _nt);
 }
 
static void _ode_matsol(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ica = _ion_ica;
  cai = _ion_cai;
  cai = _ion_cai;
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_ca_sym, _ppvar, 0, 3);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 1, 1);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  ca = ca0;
 {
   ca = cainf ;
   cai = ca ;
   }
 
}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
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
  ica = _ion_ica;
  cai = _ion_cai;
  cai = _ion_cai;
 initmodel(_p, _ppvar, _thread, _nt);
  _ion_cai = cai;
  nrn_wrote_conc(_ca_sym, (&(_ion_cai)) - 1, _style_ca);
}
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{
} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
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
 
}
 
}

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
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
 
}
 
}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
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
  ica = _ion_ica;
  cai = _ion_cai;
  cai = _ion_cai;
 {   state(_p, _ppvar, _thread, _nt);
  } {
   }
  _ion_cai = cai;
}}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = ca_columnindex;  _dlist1[0] = Dca_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/Users/katharinaduecker/Documents/Projects Brown/hnn-core/hnn_core/mod/cad.mod";
static const char* nmodl_file_text = 
  "COMMENT\n"
  "    26 Ago 2002 Modification of original channel to allow variable time step\n"
  "            and to correct an initialization error.\n"
  "\n"
  "    Done by Michael Hines (michael.hines@yale.edu) and Ruggero Scorcioni (rscorcio@gmu.edu)\n"
  "             at EU Advance Course in Computational Neuroscience. Obidos, Portugal\n"
  "\n"
  "    Internal calcium concentration due to calcium currents and pump.\n"
  "    Differential equations.\n"
  "\n"
  "    Simple model of ATPase pump with 3 kinetic constants (Destexhe 92)\n"
  "            Cai + P <-> CaP -> Cao + P  (k1,k2,k3)\n"
  "\n"
  "    A Michaelis-Menten approximation is assumed, which reduces the complexity\n"
  "            of the system to 2 parameters:\n"
  "\n"
  "            kt = <tot enzyme concentration> * k3  -> TIME CONSTANT OF THE PUMP\n"
  "            kd = k2/k1 (dissociation constant)    -> EQUILIBRIUM CALCIUM VALUE\n"
  "\n"
  "    The values of these parameters are chosen assuming a high affinity of\n"
  "        the pump to calcium and a low transport capacity (cfr. Blaustein,\n"
  "        TINS, 11: 438, 1988, and references therein).\n"
  "\n"
  "    Units checked using \"modlunit\" -> factor 10000 needed in ca entry\n"
  "\n"
  "    VERSION OF PUMP + DECAY (decay can be viewed as simplified buffering)\n"
  "\n"
  "    All variables are range variables\n"
  "\n"
  "    This mechanism was published in:  Destexhe, A. Babloyantz, A. and\n"
  "    Sejnowski, TJ.  Ionic mechanisms for intrinsic slow oscillations in\n"
  "    thalamic relay neurons. Biophys. J. 65: 1538-1552, 1993)\n"
  "\n"
  "    Written by Alain Destexhe, Salk Institute, Nov 12, 1992\n"
  "\n"
  "ENDCOMMENT\n"
  "\n"
  "TITLE Decay of internal calcium concentration\n"
  "\n"
  "INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}\n"
  "\n"
  "NEURON {\n"
  "    SUFFIX cad\n"
  "    USEION ca\n"
  "    READ ica, cai\n"
  "    WRITE cai\n"
  "\n"
  "    : SRJones put taur up here\n"
  "    RANGE ca, taur\n"
  "    GLOBAL depth, cainf\n"
  "    : GLOBAL depth, cainf, taur\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "    : moles do not appear in units\n"
  "    (molar) = (1/liter)\n"
  "    (mM)    = (millimolar)\n"
  "    (um)    = (micron)\n"
  "    (mA)    = (milliamp)\n"
  "    (msM)   = (ms mM)\n"
  "    FARADAY = (faraday) (coulomb)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "    depth = .1     (um)        : depth of shell\n"
  "    taur  = 200    (ms)        : rate of calcium removal 200 default\n"
  "    cainf = 100e-6 (mM)\n"
  "    cai            (mM)\n"
  "}\n"
  "\n"
  "STATE {\n"
  "    ca             (mM) <1e-5>\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "    ca = cainf\n"
  "    cai = ca\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "    ica            (mA/cm2)\n"
  "    drive_channel  (mM/ms)\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "    SOLVE state METHOD cnexp\n"
  "}\n"
  "\n"
  "DERIVATIVE state {\n"
  "    drive_channel = -(10000) * ica / (2 * FARADAY * depth)\n"
  "\n"
  "    : cannot pump inward\n"
  "    if (drive_channel <= 0.) { drive_channel = 0. }\n"
  "\n"
  "    ca' = drive_channel + (cainf-ca) / taur\n"
  "    cai = ca\n"
  "}\n"
  ;
#endif
