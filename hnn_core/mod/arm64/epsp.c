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
 
#define nrn_init _nrn_init__epsp
#define _nrn_initial _nrn_initial__epsp
#define nrn_cur _nrn_cur__epsp
#define _nrn_current _nrn_current__epsp
#define nrn_jacob _nrn_jacob__epsp
#define nrn_state _nrn_state__epsp
#define _net_receive _net_receive__epsp 
 
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
#define onset _p[0]
#define onset_columnindex 0
#define tau0 _p[1]
#define tau0_columnindex 1
#define tau1 _p[2]
#define tau1_columnindex 2
#define imax _p[3]
#define imax_columnindex 3
#define i _p[4]
#define i_columnindex 4
#define myv _p[5]
#define myv_columnindex 5
#define v _p[6]
#define v_columnindex 6
#define _g _p[7]
#define _g_columnindex 7
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
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 /* declaration of user functions */
 static double _hoc_curr(void*);
 static double _hoc_myexp(void*);
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
 _extcall_prop = _prop;
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
 "curr", _hoc_curr,
 "myexp", _hoc_myexp,
 0, 0
};
#define curr curr_epsp
#define myexp myexp_epsp
 extern double curr( _threadargsprotocomma_ double );
 extern double myexp( _threadargsprotocomma_ double );
 #define _za (_thread[0]._pval + 0)
 #define _ztpeak _thread[0]._pval[2]
 #define _zadjust _thread[0]._pval[3]
 #define _zamp _thread[0]._pval[4]
 /* declare global and static user variables */
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "onset", "ms",
 "tau0", "ms",
 "tau1", "ms",
 "imax", "nA",
 "i", "nA",
 "myv", "mV",
 0,0
};
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
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
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"epsp",
 "onset",
 "tau0",
 "tau1",
 "imax",
 0,
 "i",
 "myv",
 0,
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
 	_p = nrn_prop_data_alloc(_mechtype, 8, _prop);
 	/*initialize range parameters*/
 	onset = 0;
 	tau0 = 0.2;
 	tau1 = 3;
 	imax = 0;
  }
 	_prop->param = _p;
 	_prop->param_size = 8;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 2, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
 static void _thread_mem_init(Datum*);
 static void _thread_cleanup(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _epsp_reg() {
	int _vectorized = 1;
  _initlists();
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init,
	 hoc_nrnpointerindex, 2,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
  _extcall_thread = (Datum*)ecalloc(1, sizeof(Datum));
  _thread_mem_init(_extcall_thread);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 1, _thread_mem_init);
     _nrn_thread_reg(_mechtype, 0, _thread_cleanup);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 8, 2);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 epsp /Users/katharinaduecker/Documents/projects_brown/hnn-tuning/local_hnn/hnn_core/mod/epsp.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 /*Top LOCAL _za [ 2 ] */
 /*Top LOCAL _ztpeak */
 /*Top LOCAL _zadjust */
 /*Top LOCAL _zamp */
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
 
double myexp ( _threadargsprotocomma_ double _lx ) {
   double _lmyexp;
 if ( _lx < - 100.0 ) {
     _lmyexp = 0.0 ;
     }
   else {
     _lmyexp = exp ( _lx ) ;
     }
   
return _lmyexp;
 }
 
static double _hoc_myexp(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  myexp ( _p, _ppvar, _thread, _nt, *getarg(1) );
 return(_r);
}
 
double curr ( _threadargsprotocomma_ double _lx ) {
   double _lcurr;
 _ztpeak = tau0 * tau1 * log ( tau0 / tau1 ) / ( tau0 - tau1 ) ;
   _zadjust = 1.0 / ( ( 1.0 - myexp ( _threadargscomma_ - _ztpeak / tau0 ) ) - ( 1.0 - myexp ( _threadargscomma_ - _ztpeak / tau1 ) ) ) ;
   _zamp = _zadjust * imax ;
   if ( _lx < onset ) {
     _lcurr = 0.0 ;
     }
   else {
     _za [ 0 ] = 1.0 - myexp ( _threadargscomma_ - ( _lx - onset ) / tau0 ) ;
     _za [ 1 ] = 1.0 - myexp ( _threadargscomma_ - ( _lx - onset ) / tau1 ) ;
     _lcurr = - _zamp * ( _za [ 0 ] - _za [ 1 ] ) ;
     }
   
return _lcurr;
 }
 
static double _hoc_curr(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  curr ( _p, _ppvar, _thread, _nt, *getarg(1) );
 return(_r);
}
 
static void _thread_mem_init(Datum* _thread) {
   _thread[0]._pval = (double*)ecalloc(5, sizeof(double));
 }
 
static void _thread_cleanup(Datum* _thread) {
   free((void*)(_thread[0]._pval));
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{

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
 initmodel(_p, _ppvar, _thread, _nt);
}
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   myv = v ;
   i = curr ( _threadargscomma_ t ) ;
   }
 _current += i;

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
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
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

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/Users/katharinaduecker/Documents/projects_brown/hnn-tuning/local_hnn/hnn_core/mod/epsp.mod";
static const char* nmodl_file_text = 
  ": this model is built-in to neuron with suffix epsp\n"
  ": Schaefer et al. 2003\n"
  "\n"
  "COMMENT\n"
  "modified from syn2.mod\n"
  "injected current with exponential rise and decay current defined by\n"
  "         i = 0 for t < onset and\n"
  "         i=amp*((1-exp(-(t-onset)/tau0))-(1-exp(-(t-onset)/tau1)))\n"
  "          for t > onset\n"
  "\n"
  "	compare to experimental current injection:\n"
  " 	i = - amp*(1-exp(-t/t1))*(exp(-t/t2))\n"
  "\n"
  "	-> tau1==t2   tau0 ^-1 = t1^-1 + t2^-1\n"
  "ENDCOMMENT\n"
  "					       \n"
  "INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}\n"
  "\n"
  "NEURON {\n"
  "	POINT_PROCESS epsp\n"
  "	RANGE onset, tau0, tau1, imax, i, myv\n"
  "	NONSPECIFIC_CURRENT i\n"
  "}\n"
  "UNITS {\n"
  "	(nA) = (nanoamp)\n"
  "	(mV) = (millivolt)\n"
  "	(umho) = (micromho)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	onset=0  (ms)\n"
  "	tau0=0.2 (ms)\n"
  "	tau1=3.0 (ms)\n"
  "	imax=0 	 (nA)\n"
  "	v	 (mV)\n"
  "}\n"
  "\n"
  "ASSIGNED { i (nA)  myv (mV)}\n"
  "\n"
  "LOCAL   a[2]\n"
  "LOCAL   tpeak\n"
  "LOCAL   adjust\n"
  "LOCAL   amp\n"
  "\n"
  "BREAKPOINT {\n"
  "	myv = v\n"
  "        i = curr(t)\n"
  "}\n"
  "\n"
  "FUNCTION myexp(x) {\n"
  "	if (x < -100) {\n"
  "	myexp = 0\n"
  "	}else{\n"
  "	myexp = exp(x)\n"
  "	}\n"
  "}\n"
  "\n"
  "FUNCTION curr(x (ms)) (nA) {				\n"
  "	tpeak=tau0*tau1*log(tau0/tau1)/(tau0-tau1)\n"
  "	adjust=1/((1-myexp(-tpeak/tau0))-(1-myexp(-tpeak/tau1)))\n"
  "	amp=adjust*imax\n"
  "	if (x < onset) {\n"
  "		curr = 0\n"
  "	}else{\n"
  "		a[0]=1-myexp(-(x-onset)/tau0)\n"
  "		a[1]=1-myexp(-(x-onset)/tau1)\n"
  "		curr = -amp*(a[0]-a[1])\n"
  "	}\n"
  "}\n"
  ;
#endif
