#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;
#if defined(__cplusplus)
extern "C" {
#endif

extern void _ar_reg(void);
extern void _ca_reg(void);
extern void _cad_reg(void);
extern void _cat_reg(void);
extern void _dipole_reg(void);
extern void _dipole_pp_reg(void);
extern void _hh2_reg(void);
extern void _kca_reg(void);
extern void _km_reg(void);
extern void _vecevent_reg(void);

void modl_reg() {
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");
    fprintf(stderr, " \"ar.mod\"");
    fprintf(stderr, " \"ca.mod\"");
    fprintf(stderr, " \"cad.mod\"");
    fprintf(stderr, " \"cat.mod\"");
    fprintf(stderr, " \"dipole.mod\"");
    fprintf(stderr, " \"dipole_pp.mod\"");
    fprintf(stderr, " \"hh2.mod\"");
    fprintf(stderr, " \"kca.mod\"");
    fprintf(stderr, " \"km.mod\"");
    fprintf(stderr, " \"vecevent.mod\"");
    fprintf(stderr, "\n");
  }
  _ar_reg();
  _ca_reg();
  _cad_reg();
  _cat_reg();
  _dipole_reg();
  _dipole_pp_reg();
  _hh2_reg();
  _kca_reg();
  _km_reg();
  _vecevent_reg();
}

#if defined(__cplusplus)
}
#endif
