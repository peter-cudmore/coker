#ifndef COKER_OSQP_EMBEDDED_BINDINGS_WRAPPER_H
#define COKER_OSQP_EMBEDDED_BINDINGS_WRAPPER_H

#if !defined(EMBEDDED) || (EMBEDDED != 2)
#error "embedded_bindings_wrapper.h requires EMBEDDED=2"
#endif

#include "osqp.h"
#include "types.h"
#include "auxil.h"
#include "kkt.h"
#include "algebra_impl.h"
#include "qdldl_interface.h"

#endif
