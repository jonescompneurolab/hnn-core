#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;
#if defined(__cplusplus)
extern "C" {
#endif

extern void _CaDynamics_E2_reg(void);
extern void _Ca_HVA_reg(void);
extern void _Ca_LVAst_reg(void);
extern void _Ih_reg(void);
extern void _Im_reg(void);
extern void _K_Pst_reg(void);
extern void _K_Tst_reg(void);
extern void _NaTa_t_reg(void);
extern void _NaTs2_t_reg(void);
extern void _Nap_Et2_reg(void);
extern void _SK_E2_reg(void);
extern void _SKv3_1_reg(void);
extern void _ar_reg(void);
extern void _ca_reg(void);
extern void _cad_reg(void);
extern void _cat_reg(void);
extern void _dipole_reg(void);
extern void _dipole_pp_reg(void);
extern void _epsp_reg(void);
extern void _hh2_reg(void);
extern void _kca_reg(void);
extern void _kd_reg(void);
extern void _kdr_reg(void);
extern void _km_reg(void);
extern void _nas_reg(void);
extern void _vecevent_reg(void);

void modl_reg() {
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");
    fprintf(stderr, " \"CaDynamics_E2.mod\"");
    fprintf(stderr, " \"Ca_HVA.mod\"");
    fprintf(stderr, " \"Ca_LVAst.mod\"");
    fprintf(stderr, " \"Ih.mod\"");
    fprintf(stderr, " \"Im.mod\"");
    fprintf(stderr, " \"K_Pst.mod\"");
    fprintf(stderr, " \"K_Tst.mod\"");
    fprintf(stderr, " \"NaTa_t.mod\"");
    fprintf(stderr, " \"NaTs2_t.mod\"");
    fprintf(stderr, " \"Nap_Et2.mod\"");
    fprintf(stderr, " \"SK_E2.mod\"");
    fprintf(stderr, " \"SKv3_1.mod\"");
    fprintf(stderr, " \"ar.mod\"");
    fprintf(stderr, " \"ca.mod\"");
    fprintf(stderr, " \"cad.mod\"");
    fprintf(stderr, " \"cat.mod\"");
    fprintf(stderr, " \"dipole.mod\"");
    fprintf(stderr, " \"dipole_pp.mod\"");
    fprintf(stderr, " \"epsp.mod\"");
    fprintf(stderr, " \"hh2.mod\"");
    fprintf(stderr, " \"kca.mod\"");
    fprintf(stderr, " \"kd.mod\"");
    fprintf(stderr, " \"kdr.mod\"");
    fprintf(stderr, " \"km.mod\"");
    fprintf(stderr, " \"nas.mod\"");
    fprintf(stderr, " \"vecevent.mod\"");
    fprintf(stderr, "\n");
  }
  _CaDynamics_E2_reg();
  _Ca_HVA_reg();
  _Ca_LVAst_reg();
  _Ih_reg();
  _Im_reg();
  _K_Pst_reg();
  _K_Tst_reg();
  _NaTa_t_reg();
  _NaTs2_t_reg();
  _Nap_Et2_reg();
  _SK_E2_reg();
  _SKv3_1_reg();
  _ar_reg();
  _ca_reg();
  _cad_reg();
  _cat_reg();
  _dipole_reg();
  _dipole_pp_reg();
  _epsp_reg();
  _hh2_reg();
  _kca_reg();
  _kd_reg();
  _kdr_reg();
  _km_reg();
  _nas_reg();
  _vecevent_reg();
}

#if defined(__cplusplus)
}
#endif
