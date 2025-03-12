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
 
#define nrn_init _nrn_init__GABAB
#define _nrn_initial _nrn_initial__GABAB
#define nrn_cur _nrn_cur__GABAB
#define _nrn_current _nrn_current__GABAB
#define nrn_jacob _nrn_jacob__GABAB
#define nrn_state _nrn_state__GABAB
#define _net_receive _net_receive__GABAB 
#define bindkin bindkin__GABAB 
 
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
#define i _p[0]
#define i_columnindex 0
#define g _p[1]
#define g_columnindex 1
#define R _p[2]
#define R_columnindex 2
#define Ron _p[3]
#define Ron_columnindex 3
#define Roff _p[4]
#define Roff_columnindex 4
#define G _p[5]
#define G_columnindex 5
#define Gn _p[6]
#define Gn_columnindex 6
#define edc _p[7]
#define edc_columnindex 7
#define synon _p[8]
#define synon_columnindex 8
#define Rinf _p[9]
#define Rinf_columnindex 9
#define Rtau _p[10]
#define Rtau_columnindex 10
#define Beta _p[11]
#define Beta_columnindex 11
#define DRon _p[12]
#define DRon_columnindex 12
#define DRoff _p[13]
#define DRoff_columnindex 13
#define DG _p[14]
#define DG_columnindex 14
#define _g _p[15]
#define _g_columnindex 15
#define _tsav _p[16]
#define _tsav_columnindex 16
#define _nd_area  *_ppvar[0]._pval
 
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

 extern Prop* nrn_point_prop_;
 static int _pointtype;
 static void* _hoc_create_pnt(Object* _ho) { void* create_point_process(int, Object*);
 return create_point_process(_pointtype, _ho);
}
 static void _hoc_destroy_pnt(void*);
 static double _hoc_loc_pnt(void* _vptr) {double loc_point_process(int, void*);
 return loc_point_process(_pointtype, _vptr);
}
 static double _hoc_has_loc(void* _vptr) {double has_loc_point(void*);
 return has_loc_point(_vptr);
}
 static double _hoc_get_loc_pnt(void* _vptr) {
 double get_loc_point_process(void*); return (get_loc_point_process(_vptr));
}
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _p = _prop->param; _ppvar = _prop->dparam;
 }
 static void _hoc_setdata(void* _vptr) { Prop* _prop;
 _prop = ((Point_process*)_vptr)->_prop;
   _setdata(_prop);
 }
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 0,0
};
 static Member_func _member_func[] = {
 "loc", _hoc_loc_pnt,
 "has_loc", _hoc_has_loc,
 "get_loc", _hoc_get_loc_pnt,
 0, 0
};
 /* declare global and static user variables */
#define Cdur Cdur_GABAB
 double Cdur = 0.3;
#define Cmax Cmax_GABAB
 double Cmax = 0.5;
#define Erev Erev_GABAB
 double Erev = -95;
#define KD KD_GABAB
 double KD = 100;
#define K4 K4_GABAB
 double K4 = 0.033;
#define K3 K3_GABAB
 double K3 = 0.098;
#define K2 K2_GABAB
 double K2 = 0.0013;
#define K1 K1_GABAB
 double K1 = 0.52;
#define cutoff cutoff_GABAB
 double cutoff = 1e+12;
#define n n_GABAB
 double n = 4;
#define warn warn_GABAB
 double warn = 0;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "Cmax_GABAB", "mM",
 "Cdur_GABAB", "ms",
 "K1_GABAB", "/ms",
 "K2_GABAB", "/ms",
 "K3_GABAB", "/ms",
 "K4_GABAB", "/ms",
 "Erev_GABAB", "mV",
 "i", "nA",
 "g", "umho",
 0,0
};
 static double G0 = 0;
 static double Roff0 = 0;
 static double Ron0 = 0;
 static double delta_t = 1;
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "Cmax_GABAB", &Cmax_GABAB,
 "Cdur_GABAB", &Cdur_GABAB,
 "K1_GABAB", &K1_GABAB,
 "K2_GABAB", &K2_GABAB,
 "K3_GABAB", &K3_GABAB,
 "K4_GABAB", &K4_GABAB,
 "KD_GABAB", &KD_GABAB,
 "n_GABAB", &n_GABAB,
 "Erev_GABAB", &Erev_GABAB,
 "warn_GABAB", &warn_GABAB,
 "cutoff_GABAB", &cutoff_GABAB,
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
 static void _hoc_destroy_pnt(void* _vptr) {
   destroy_point_process(_vptr);
}
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(NrnThread*, _Memb_list*, int);
static void _ode_matsol(NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[3]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"GABAB",
 0,
 "i",
 "g",
 "R",
 0,
 "Ron",
 "Roff",
 "G",
 0,
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
  if (nrn_point_prop_) {
	_prop->_alloc_seq = nrn_point_prop_->_alloc_seq;
	_p = nrn_point_prop_->param;
	_ppvar = nrn_point_prop_->dparam;
 }else{
 	_p = nrn_prop_data_alloc(_mechtype, 17, _prop);
 	/*initialize range parameters*/
  }
 	_prop->param = _p;
 	_prop->param_size = 17;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 
#define _tqitem &(_ppvar[2]._pvoid)
 static void _net_receive(Point_process*, double*, double);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _gabab_reg() {
	int _vectorized = 0;
  _initlists();
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init,
	 hoc_nrnpointerindex, 0,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 17, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "netsend");
  hoc_register_dparam_semantics(_mechtype, 3, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_size[_mechtype] = 3;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 GABAB /Users/katharinaduecker/Documents/projects_brown/hnn-tuning/local_hnn/hnn_core/mod/gabab.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
 static int _deriv1_advance = 0;
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist2[3]; static double _dlist2[3];
 static double _savstate1[3], *_temp1 = _savstate1;
 static int _slist1[3], _dlist1[3];
 static int bindkin(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 () {_reset=0;
 {
   DRon = synon * K1 * Cmax - ( K1 * Cmax + K2 ) * Ron ;
   DRoff = - K2 * Roff ;
   R = Ron + Roff ;
   DG = K3 * R - K4 * G ;
   }
 return _reset;
}
 static int _ode_matsol1 () {
 DRon = DRon  / (1. - dt*( ( - ( ( K1 * Cmax + K2 ) )*( 1.0 ) ) )) ;
 DRoff = DRoff  / (1. - dt*( ( - K2 )*( 1.0 ) )) ;
 R = Ron + Roff ;
 DG = DG  / (1. - dt*( ( - ( K4 )*( 1.0 ) ) )) ;
  return 0;
}
 /*END CVODE*/
 
static int bindkin () {_reset=0;
 { static int _recurse = 0;
 int _counte = -1;
 if (!_recurse) {
 _recurse = 1;
 {int _id; for(_id=0; _id < 3; _id++) { _savstate1[_id] = _p[_slist1[_id]];}}
 error = newton(3,_slist2, _p, bindkin, _dlist2);
 _recurse = 0; if(error) {abort_run(error);}}
 {
   DRon = synon * K1 * Cmax - ( K1 * Cmax + K2 ) * Ron ;
   DRoff = - K2 * Roff ;
   R = Ron + Roff ;
   DG = K3 * R - K4 * G ;
   {int _id; for(_id=0; _id < 3; _id++) {
if (_deriv1_advance) {
 _dlist2[++_counte] = _p[_dlist1[_id]] - (_p[_slist1[_id]] - _savstate1[_id])/dt;
 }else{
_dlist2[++_counte] = _p[_slist1[_id]] - _savstate1[_id];}}}
 } }
 return _reset;}
 
static void _net_receive (Point_process* _pnt, double* _args, double _lflag) 
{    _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
  if (_tsav > t){ extern char* hoc_object_name(); hoc_execerror(hoc_object_name(_pnt->ob), ":Event arrived out of order. Must call ParallelContext.set_maxstep AFTER assigning minimum NetCon.delay");}
 _tsav = t;   if (_lflag == 1. ) {*(_tqitem) = 0;}
 {
   if ( _lflag  == 1.0 ) {
     _args[1] = _args[0] * ( Rinf + ( _args[1] - Rinf ) * exp ( - ( t - _args[2] ) / Rtau ) ) ;
     _args[2] = t ;
     synon = synon - _args[0] ;
         if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for general derivimplicit and KINETIC case */
    int __i, __neq = 3;
    double __state = Ron;
    double __primary_delta = (Ron - _args[1] ) - __state;
    double __dtsav = dt;
    for (__i = 0; __i < __neq; ++__i) {
      _p[_dlist1[__i]] = 0.0;
    }
    _p[_dlist1[0]] = __primary_delta;
    dt *= 0.5;
    v = NODEV(_pnt->node);
#if NRN_VECTORIZED
    _thread = _nt->_ml_list[_mechtype]->_thread;
#endif
    _ode_matsol_instance1(_threadargs_);
    dt = __dtsav;
    for (__i = 0; __i < __neq; ++__i) {
      _p[_slist1[__i]] += _p[_dlist1[__i]];
    }
  } else {
 Ron = Ron - _args[1]  ;
       }
     if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for general derivimplicit and KINETIC case */
    int __i, __neq = 3;
    double __state = Roff;
    double __primary_delta = (Roff + _args[1] ) - __state;
    double __dtsav = dt;
    for (__i = 0; __i < __neq; ++__i) {
      _p[_dlist1[__i]] = 0.0;
    }
    _p[_dlist1[1]] = __primary_delta;
    dt *= 0.5;
    v = NODEV(_pnt->node);
#if NRN_VECTORIZED
    _thread = _nt->_ml_list[_mechtype]->_thread;
#endif
    _ode_matsol_instance1(_threadargs_);
    dt = __dtsav;
    for (__i = 0; __i < __neq; ++__i) {
      _p[_slist1[__i]] += _p[_dlist1[__i]];
    }
  } else {
 Roff = Roff + _args[1]  ;
       }
 }
   else {
     _args[1] = _args[0] * _args[1] * exp ( - Beta * ( t - _args[2] ) ) ;
     _args[2] = t ;
     synon = synon + _args[0] ;
         if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for general derivimplicit and KINETIC case */
    int __i, __neq = 3;
    double __state = Ron;
    double __primary_delta = (Ron + _args[1] ) - __state;
    double __dtsav = dt;
    for (__i = 0; __i < __neq; ++__i) {
      _p[_dlist1[__i]] = 0.0;
    }
    _p[_dlist1[0]] = __primary_delta;
    dt *= 0.5;
    v = NODEV(_pnt->node);
#if NRN_VECTORIZED
    _thread = _nt->_ml_list[_mechtype]->_thread;
#endif
    _ode_matsol_instance1(_threadargs_);
    dt = __dtsav;
    for (__i = 0; __i < __neq; ++__i) {
      _p[_slist1[__i]] += _p[_dlist1[__i]];
    }
  } else {
 Ron = Ron + _args[1]  ;
       }
     if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for general derivimplicit and KINETIC case */
    int __i, __neq = 3;
    double __state = Roff;
    double __primary_delta = (Roff - _args[1] ) - __state;
    double __dtsav = dt;
    for (__i = 0; __i < __neq; ++__i) {
      _p[_dlist1[__i]] = 0.0;
    }
    _p[_dlist1[1]] = __primary_delta;
    dt *= 0.5;
    v = NODEV(_pnt->node);
#if NRN_VECTORIZED
    _thread = _nt->_ml_list[_mechtype]->_thread;
#endif
    _ode_matsol_instance1(_threadargs_);
    dt = __dtsav;
    for (__i = 0; __i < __neq; ++__i) {
      _p[_slist1[__i]] += _p[_dlist1[__i]];
    }
  } else {
 Roff = Roff - _args[1]  ;
       }
 net_send ( _tqitem, _args, _pnt, t +  Cdur , 1.0 ) ;
     }
   } }
 
static int _ode_count(int _type){ return 3;}
 
static void _ode_spec(NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
     _ode_spec1 ();
 }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 3; ++_i) {
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
 _ode_matsol_instance1(_threadargs_);
 }}

static void initmodel() {
  int _i; double _save;_ninits++;
 _save = t;
 t = 0.0;
{
  G = G0;
  Roff = Roff0;
  Ron = Ron0;
 {
   R = 0.0 ;
   G = 0.0 ;
   Ron = 0.0 ;
   Roff = 0.0 ;
   synon = 0.0 ;
   Rinf = K1 * Cmax / ( K1 * Cmax + K2 ) ;
   Rtau = 1.0 / ( K1 * Cmax + K2 ) ;
   Beta = K2 ;
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
 _tsav = -1e20;
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
 initmodel();
}}

static double _nrn_current(double _v){double _current=0.;v=_v;{ {
   if ( G < cutoff ) {
     Gn = G * G * G * G ;
     g = Gn / ( Gn + KD ) ;
     }
   else {
     if (  ! warn ) {
       printf ( "gabab.mod WARN: G = %g too large\n" , G ) ;
       warn = 1.0 ;
       }
     g = 1.0 ;
     }
   i = g * ( v - Erev ) ;
   }
 _current += i;

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
 _g = _nrn_current(_v + .001);
 	{ _rhs = _nrn_current(_v);
 	}
 _g = (_g - _rhs)/.001;
 _g *=  1.e2/(_nd_area);
 _rhs *= 1.e2/(_nd_area);
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
double _dtsav = dt;
if (secondorder) { dt *= 0.5; }
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
 { error = _deriv1_advance = 1;
 derivimplicit(_ninits, 3, _slist1, _dlist1, _p, &t, dt, bindkin, &_temp1);
_deriv1_advance = 0;
 if(error){fprintf(stderr,"at line 168 in file gabab.mod:\n	SOLVE bindkin METHOD derivimplicit\n"); nrn_complain(_p); abort_run(error);}
    if (secondorder) {
    int _i;
    for (_i = 0; _i < 3; ++_i) {
      _p[_slist1[_i]] += dt*_p[_dlist1[_i]];
    }}
 }}}
 dt = _dtsav;
}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = Ron_columnindex;  _dlist1[0] = DRon_columnindex;
 _slist1[1] = Roff_columnindex;  _dlist1[1] = DRoff_columnindex;
 _slist1[2] = G_columnindex;  _dlist1[2] = DG_columnindex;
 _slist2[0] = G_columnindex;
 _slist2[1] = Roff_columnindex;
 _slist2[2] = Ron_columnindex;
_first = 0;
}

#if NMODL_TEXT
static const char* nmodl_filename = "/Users/katharinaduecker/Documents/projects_brown/hnn-tuning/local_hnn/hnn_core/mod/gabab.mod";
static const char* nmodl_file_text = 
  ": $Id: gabab.mod,v 1.9 2004/06/17 16:04:05 billl Exp $\n"
  "\n"
  "COMMENT\n"
  "-----------------------------------------------------------------------------\n"
  "\n"
  "	Kinetic model of GABA-B receptors\n"
  "	=================================\n"
  "\n"
  "  MODEL OF SECOND-ORDER G-PROTEIN TRANSDUCTION AND FAST K+ OPENING\n"
  "  WITH COOPERATIVITY OF G-PROTEIN BINDING TO K+ CHANNEL\n"
  "\n"
  "  PULSE OF TRANSMITTER\n"
  "\n"
  "  SIMPLE KINETICS WITH NO DESENSITIZATION\n"
  "\n"
  "	Features:\n"
  "\n"
  "  	  - peak at 100 ms; time course fit to Tom Otis' PSC\n"
  "	  - SUMMATION (psc is much stronger with bursts)\n"
  "\n"
  "\n"
  "	Approximations:\n"
  "\n"
  "	  - single binding site on receptor	\n"
  "	  - model of alpha G-protein activation (direct) of K+ channel\n"
  "	  - G-protein dynamics is second-order; simplified as follows:\n"
  "		- saturating receptor\n"
  "		- no desensitization\n"
  "		- Michaelis-Menten of receptor for G-protein production\n"
  "		- \"resting\" G-protein is in excess\n"
  "		- Quasi-stat of intermediate enzymatic forms\n"
  "	  - binding on K+ channel is fast\n"
  "\n"
  "\n"
  "	Kinetic Equations:\n"
  "\n"
  "	  dR/dt = K1 * T * (1-R-D) - K2 * R\n"
  "\n"
  "	  dG/dt = K3 * R - K4 * G\n"
  "\n"
  "	  R : activated receptor\n"
  "	  T : transmitter\n"
  "	  G : activated G-protein\n"
  "	  K1,K2,K3,K4 = kinetic rate cst\n"
  "\n"
  "  n activated G-protein bind to a K+ channel:\n"
  "\n"
  "	n G + C <-> O		(Alpha,Beta)\n"
  "\n"
  "  If the binding is fast, the fraction of open channels is given by:\n"
  "\n"
  "	O = G^n / ( G^n + KD )\n"
  "\n"
  "  where KD = Beta / Alpha is the dissociation constant\n"
  "\n"
  "-----------------------------------------------------------------------------\n"
  "\n"
  "  Parameters estimated from patch clamp recordings of GABAB PSP's in\n"
  "  rat hippocampal slices (Otis et al, J. Physiol. 463: 391-407, 1993).\n"
  "\n"
  "-----------------------------------------------------------------------------\n"
  "\n"
  "  PULSE MECHANISM\n"
  "\n"
  "  Kinetic synapse with release mechanism as a pulse.  \n"
  "\n"
  "  Warning: for this mechanism to be equivalent to the model with diffusion \n"
  "  of transmitter, small pulses must be used...\n"
  "\n"
  "  For a detailed model of GABAB:\n"
  "\n"
  "  Destexhe, A. and Sejnowski, T.J.  G-protein activation kinetics and\n"
  "  spill-over of GABA may account for differences between inhibitory responses\n"
  "  in the hippocampus and thalamus.  Proc. Natl. Acad. Sci. USA  92:\n"
  "  9515-9519, 1995.\n"
  "\n"
  "  For a review of models of synaptic currents:\n"
  "\n"
  "  Destexhe, A., Mainen, Z.F. and Sejnowski, T.J.  Kinetic models of \n"
  "  synaptic transmission.  In: Methods in Neuronal Modeling (2nd edition; \n"
  "  edited by Koch, C. and Segev, I.), MIT press, Cambridge, 1996.\n"
  "\n"
  "  This simplified model was introduced in:\n"
  "\n"
  "  Destexhe, A., Bal, T., McCormick, D.A. and Sejnowski, T.J.\n"
  "  Ionic mechanisms underlying synchronized oscillations and propagating\n"
  "  waves in a model of ferret thalamic slices. Journal of Neurophysiology\n"
  "  76: 2049-2070, 1996.  \n"
  "\n"
  "  See also http://www.cnl.salk.edu/~alain\n"
  "\n"
  "\n"
  "\n"
  "  Alain Destexhe, Salk Institute and Laval University, 1995\n"
  "\n"
  "-----------------------------------------------------------------------------\n"
  "ENDCOMMENT\n"
  "\n"
  "\n"
  "\n"
  "INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}\n"
  "\n"
  "NEURON {\n"
  "	POINT_PROCESS GABAB\n"
  "	RANGE R, G, g\n"
  "	NONSPECIFIC_CURRENT i\n"
  "	GLOBAL Cmax, Cdur\n"
  "	GLOBAL K1, K2, K3, K4, KD, Erev, warn, cutoff\n"
  "}\n"
  "UNITS {\n"
  "	(nA) = (nanoamp)\n"
  "	(mV) = (millivolt)\n"
  "	(umho) = (micromho)\n"
  "	(mM) = (milli/liter)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "\n"
  "	Cmax	= 0.5	(mM)		: max transmitter concentration\n"
  "	Cdur	= 0.3	(ms)		: transmitter duration (rising phase)\n"
  ":\n"
  ":	From Kfit with long pulse (5ms 0.5mM)\n"
  ":\n"
  "	K1	= 0.52	(/ms mM)	: forward binding rate to receptor\n"
  "	K2	= 0.0013 (/ms)		: backward (unbinding) rate of receptor\n"
  "	K3	= 0.098 (/ms)		: rate of G-protein production\n"
  "	K4	= 0.033 (/ms)		: rate of G-protein decay\n"
  "	KD	= 100			: dissociation constant of K+ channel\n"
  "	n	= 4			: nb of binding sites of G-protein on K+\n"
  "	Erev	= -95	(mV)		: reversal potential (E_K)\n"
  "	warn	= 0			: too large G warning has/has not been issued\n"
  "        cutoff = 1e12\n"
  "}\n"
  "\n"
  "\n"
  "ASSIGNED {\n"
  "	v		(mV)		: postsynaptic voltage\n"
  "	i 		(nA)		: current = g*(v - Erev)\n"
  "	g 		(umho)		: conductance\n"
  "	Gn\n"
  "	R				: fraction of activated receptor\n"
  "	edc\n"
  "	synon\n"
  "	Rinf\n"
  "	Rtau (ms)\n"
  "	Beta (/ms)\n"
  "}\n"
  "\n"
  "STATE {\n"
  "	Ron Roff\n"
  "	G				: fraction of activated G-protein\n"
  "}\n"
  "\n"
  "\n"
  "INITIAL {\n"
  "	R = 0\n"
  "	G = 0\n"
  "	Ron = 0\n"
  "	Roff = 0\n"
  "	synon = 0\n"
  "	Rinf = K1*Cmax/(K1*Cmax + K2)\n"
  "	Rtau = 1/(K1*Cmax + K2)\n"
  "	Beta = K2\n"
  "\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE bindkin METHOD derivimplicit\n"
  "	if (G < cutoff) {\n"
  "		Gn = G*G*G*G : ^n = 4\n"
  "		g = Gn / (Gn+KD)\n"
  "	} else {\n"
  "		if(!warn){\n"
  "			printf(\"gabab.mod WARN: G = %g too large\\n\", G)		\n"
  "			warn = 1\n"
  "		}\n"
  "		g = 1\n"
  "	}\n"
  "	i = g*(v - Erev)\n"
  "}\n"
  "\n"
  "\n"
  "DERIVATIVE bindkin {\n"
  "	Ron' = synon*K1*Cmax - (K1*Cmax + K2)*Ron\n"
  "	Roff' = -K2*Roff\n"
  "	R = Ron + Roff\n"
  "	G' = K3 * R - K4 * G\n"
  "}\n"
  "\n"
  ": following supports both saturation from single input and\n"
  ": summation from multiple inputs\n"
  ": Note: automatic initialization of all reference args to 0 except first\n"
  "\n"
  "NET_RECEIVE(weight,  r0, t0 (ms)) {\n"
  "	if (flag == 1) { : at end of Cdur pulse so turn off\n"
  "		r0 = weight*(Rinf + (r0 - Rinf)*exp(-(t - t0)/Rtau))\n"
  "		t0 = t\n"
  "		synon = synon - weight\n"
  "		state_discontinuity(Ron, Ron - r0)\n"
  "		state_discontinuity(Roff, Roff + r0)\n"
  "        }else{ : at beginning of Cdur pulse so turn on\n"
  "		r0 = weight*r0*exp(-Beta*(t - t0))\n"
  "		t0 = t\n"
  "		synon = synon + weight\n"
  "		state_discontinuity(Ron, Ron + r0)\n"
  "		state_discontinuity(Roff, Roff - r0)\n"
  "		:come again in Cdur\n"
  "		net_send(Cdur, 1)\n"
  "        }\n"
  "}\n"
  ;
#endif
