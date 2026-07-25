#ifndef COKER_OSQP_ABI_H
#define COKER_OSQP_ABI_H
#if defined(EMBEDDED) && (EMBEDDED == 2)

#include <stddef.h>
#include <stdint.h>



#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned long coker_osqp_size_t;
enum {
    COKER_OSQP_PLAN_ABI_VERSION = 1u,
};


typedef struct {
    void *ptr;
    coker_osqp_size_t bytes;
    coker_osqp_size_t alignment;
} CokerOsqpBufferRegion;

typedef struct {
    coker_osqp_size_t bytes;
    coker_osqp_size_t alignment;
} CokerOsqpLayoutRegion;

typedef struct {
    const int32_t *col_ptr;
    const int32_t *row_idx;
    int32_t nnz;
} CokerOsqpCscView;

typedef struct {
    int32_t n;
    int32_t m;
    CokerOsqpCscView p;
    CokerOsqpCscView a;
} CokerOsqpProblemShape;

typedef struct {
    const float *p_x;
    int32_t p_nnz;
    const float *a_x;
    int32_t a_nnz;
    const float *q;
    int32_t q_len;
    const float *l;
    int32_t l_len;
    const float *u;
    int32_t u_len;
} CokerOsqpNumericUpdate;

typedef struct {
    CokerOsqpBufferRegion pdata_p;
    CokerOsqpBufferRegion pdata_i;
    CokerOsqpBufferRegion pdata_x;
    CokerOsqpBufferRegion pdata;
    CokerOsqpBufferRegion adata_p;
    CokerOsqpBufferRegion adata_i;
    CokerOsqpBufferRegion adata_x;
    CokerOsqpBufferRegion adata;
    CokerOsqpBufferRegion qdata;
    CokerOsqpBufferRegion ldata;
    CokerOsqpBufferRegion udata;
    CokerOsqpBufferRegion data;
    CokerOsqpBufferRegion settings;
    CokerOsqpBufferRegion scaling;
    CokerOsqpBufferRegion xsolution;
    CokerOsqpBufferRegion ysolution;
    CokerOsqpBufferRegion solution;
    CokerOsqpBufferRegion info;
    CokerOsqpBufferRegion qdldl_L;
    CokerOsqpBufferRegion qdldl_L_p;
    CokerOsqpBufferRegion qdldl_L_i;
    CokerOsqpBufferRegion qdldl_L_x;
    CokerOsqpBufferRegion qdldl_KKT;
    CokerOsqpBufferRegion qdldl_KKT_p;
    CokerOsqpBufferRegion qdldl_KKT_i;
    CokerOsqpBufferRegion qdldl_KKT_x;
    CokerOsqpBufferRegion qdldl;
    CokerOsqpBufferRegion qdldl_Dinv;
    CokerOsqpBufferRegion qdldl_P;
    CokerOsqpBufferRegion qdldl_bp;
    CokerOsqpBufferRegion qdldl_sol;
    CokerOsqpBufferRegion qdldl_rho_inv_vec;
    CokerOsqpBufferRegion qdldl_Pdiag_idx;
    CokerOsqpBufferRegion qdldl_PtoKKT;
    CokerOsqpBufferRegion qdldl_AtoKKT;
    CokerOsqpBufferRegion qdldl_rhotoKKT;
    CokerOsqpBufferRegion qdldl_D;
    CokerOsqpBufferRegion qdldl_etree;
    CokerOsqpBufferRegion qdldl_Lnz;
    CokerOsqpBufferRegion qdldl_iwork;
    CokerOsqpBufferRegion qdldl_bwork;
    CokerOsqpBufferRegion qdldl_fwork;
    CokerOsqpBufferRegion work_rho_vec;
    CokerOsqpBufferRegion work_rho_inv_vec;
    CokerOsqpBufferRegion work_constr_type;
    CokerOsqpBufferRegion work_x;
    CokerOsqpBufferRegion work_y;
    CokerOsqpBufferRegion work_z;
    CokerOsqpBufferRegion work_xz_tilde;
    CokerOsqpBufferRegion work_x_prev;
    CokerOsqpBufferRegion work_z_prev;
    CokerOsqpBufferRegion work_Ax;
    CokerOsqpBufferRegion work_Px;
    CokerOsqpBufferRegion work_Aty;
    CokerOsqpBufferRegion work_delta_y;
    CokerOsqpBufferRegion work_Atdelta_y;
    CokerOsqpBufferRegion work_delta_x;
    CokerOsqpBufferRegion work_Pdelta_x;
    CokerOsqpBufferRegion work_Adelta_x;
    CokerOsqpBufferRegion work_D_temp;
    CokerOsqpBufferRegion work_D_temp_A;
    CokerOsqpBufferRegion work_E_temp;
    CokerOsqpBufferRegion workspace;
} CokerOsqpBuffers;

typedef struct {
    CokerOsqpLayoutRegion pdata_p;
    CokerOsqpLayoutRegion pdata_i;
    CokerOsqpLayoutRegion pdata_x;
    CokerOsqpLayoutRegion pdata;
    CokerOsqpLayoutRegion adata_p;
    CokerOsqpLayoutRegion adata_i;
    CokerOsqpLayoutRegion adata_x;
    CokerOsqpLayoutRegion adata;
    CokerOsqpLayoutRegion qdata;
    CokerOsqpLayoutRegion ldata;
    CokerOsqpLayoutRegion udata;
    CokerOsqpLayoutRegion data;
    CokerOsqpLayoutRegion settings;
    CokerOsqpLayoutRegion scaling;
    CokerOsqpLayoutRegion xsolution;
    CokerOsqpLayoutRegion ysolution;
    CokerOsqpLayoutRegion solution;
    CokerOsqpLayoutRegion info;
    CokerOsqpLayoutRegion qdldl_L;
    CokerOsqpLayoutRegion qdldl_L_p;
    CokerOsqpLayoutRegion qdldl_L_i;
    CokerOsqpLayoutRegion qdldl_L_x;
    CokerOsqpLayoutRegion qdldl_KKT;
    CokerOsqpLayoutRegion qdldl_KKT_p;
    CokerOsqpLayoutRegion qdldl_KKT_i;
    CokerOsqpLayoutRegion qdldl_KKT_x;
    CokerOsqpLayoutRegion qdldl;
    CokerOsqpLayoutRegion qdldl_Dinv;
    CokerOsqpLayoutRegion qdldl_P;
    CokerOsqpLayoutRegion qdldl_bp;
    CokerOsqpLayoutRegion qdldl_sol;
    CokerOsqpLayoutRegion qdldl_rho_inv_vec;
    CokerOsqpLayoutRegion qdldl_Pdiag_idx;
    CokerOsqpLayoutRegion qdldl_PtoKKT;
    CokerOsqpLayoutRegion qdldl_AtoKKT;
    CokerOsqpLayoutRegion qdldl_rhotoKKT;
    CokerOsqpLayoutRegion qdldl_D;
    CokerOsqpLayoutRegion qdldl_etree;
    CokerOsqpLayoutRegion qdldl_Lnz;
    CokerOsqpLayoutRegion qdldl_iwork;
    CokerOsqpLayoutRegion qdldl_bwork;
    CokerOsqpLayoutRegion qdldl_fwork;
    CokerOsqpLayoutRegion work_rho_vec;
    CokerOsqpLayoutRegion work_rho_inv_vec;
    CokerOsqpLayoutRegion work_constr_type;
    CokerOsqpLayoutRegion work_x;
    CokerOsqpLayoutRegion work_y;
    CokerOsqpLayoutRegion work_z;
    CokerOsqpLayoutRegion work_xz_tilde;
    CokerOsqpLayoutRegion work_x_prev;
    CokerOsqpLayoutRegion work_z_prev;
    CokerOsqpLayoutRegion work_Ax;
    CokerOsqpLayoutRegion work_Px;
    CokerOsqpLayoutRegion work_Aty;
    CokerOsqpLayoutRegion work_delta_y;
    CokerOsqpLayoutRegion work_Atdelta_y;
    CokerOsqpLayoutRegion work_delta_x;
    CokerOsqpLayoutRegion work_Pdelta_x;
    CokerOsqpLayoutRegion work_Adelta_x;
    CokerOsqpLayoutRegion work_D_temp;
    CokerOsqpLayoutRegion work_D_temp_A;
    CokerOsqpLayoutRegion work_E_temp;
    CokerOsqpLayoutRegion workspace;
} CokerOsqpLayout;
typedef struct {
    const int32_t *ptr;
    int32_t len;
} CokerOsqpIndexView;

typedef struct {
    coker_osqp_size_t offset;
    coker_osqp_size_t bytes;
    coker_osqp_size_t alignment;
} CokerOsqpArenaRegion;

typedef struct {
    coker_osqp_size_t bytes;
    coker_osqp_size_t alignment;
    CokerOsqpArenaRegion pdata_x;
    CokerOsqpArenaRegion pdata;
    CokerOsqpArenaRegion adata_x;
    CokerOsqpArenaRegion adata;
    CokerOsqpArenaRegion qdata;
    CokerOsqpArenaRegion ldata;
    CokerOsqpArenaRegion udata;
    CokerOsqpArenaRegion data;
    CokerOsqpArenaRegion settings;
    CokerOsqpArenaRegion xsolution;
    CokerOsqpArenaRegion ysolution;
    CokerOsqpArenaRegion solution;
    CokerOsqpArenaRegion info;
    CokerOsqpArenaRegion qdldl_L_x;
    CokerOsqpArenaRegion qdldl_L;
    CokerOsqpArenaRegion qdldl_KKT_x;
    CokerOsqpArenaRegion qdldl_KKT;
    CokerOsqpArenaRegion qdldl;
    CokerOsqpArenaRegion qdldl_Dinv;
    CokerOsqpArenaRegion qdldl_bp;
    CokerOsqpArenaRegion qdldl_sol;
    CokerOsqpArenaRegion qdldl_rho_inv_vec;
    CokerOsqpArenaRegion qdldl_D;
    CokerOsqpArenaRegion qdldl_iwork;
    CokerOsqpArenaRegion qdldl_bwork;
    CokerOsqpArenaRegion qdldl_fwork;
    CokerOsqpArenaRegion work_rho_vec;
    CokerOsqpArenaRegion work_rho_inv_vec;
    CokerOsqpArenaRegion work_constr_type;
    CokerOsqpArenaRegion work_x;
    CokerOsqpArenaRegion work_y;
    CokerOsqpArenaRegion work_z;
    CokerOsqpArenaRegion work_xz_tilde;
    CokerOsqpArenaRegion work_x_prev;
    CokerOsqpArenaRegion work_z_prev;
    CokerOsqpArenaRegion work_Ax;
    CokerOsqpArenaRegion work_Px;
    CokerOsqpArenaRegion work_Aty;
    CokerOsqpArenaRegion work_delta_y;
    CokerOsqpArenaRegion work_Atdelta_y;
    CokerOsqpArenaRegion work_delta_x;
    CokerOsqpArenaRegion work_Pdelta_x;
    CokerOsqpArenaRegion work_Adelta_x;
    CokerOsqpArenaRegion workspace;
} CokerOsqpArenaLayout;

typedef struct {
    void *base;
    coker_osqp_size_t bytes;
    coker_osqp_size_t alignment;
} CokerOsqpArena;

typedef struct {
    float rho;
    float sigma;
    int32_t scaling;
    int32_t adaptive_rho;
    int32_t adaptive_rho_interval;
    float adaptive_rho_tolerance;
    int32_t max_iter;
    float eps_abs;
    float eps_rel;
    float eps_prim_inf;
    float eps_dual_inf;
    float alpha;
    uint32_t linsys_solver;
    int32_t scaled_termination;
    int32_t check_termination;
    int32_t warm_start;
} CokerOsqpSettings;

typedef struct {
    uint32_t abi_version;
    int32_t n;
    int32_t m;
    int32_t n_plus_m;
    CokerOsqpCscView p;
    CokerOsqpCscView a;
    CokerOsqpCscView kkt;
    CokerOsqpCscView qdldl_l;
    CokerOsqpIndexView p_to_kkt;
    CokerOsqpIndexView a_to_kkt;
    CokerOsqpIndexView rho_to_kkt;
    CokerOsqpIndexView p_diagonal_idx;
    CokerOsqpIndexView permutation;
    CokerOsqpIndexView qdldl_etree;
    CokerOsqpIndexView qdldl_lnz;
    CokerOsqpSettings settings;
    CokerOsqpArenaLayout arena_layout;
} CokerOsqpPlan;


typedef struct {
    void *pdata;
    void *adata;
    void *data;
    void *settings;
    void *scaling;
    void *solution;
    void *info;
    void *linsys_solver;
    void *qdldl;
    void *workspace;
} CokerOsqpInstance;

typedef int32_t CokerOsqpStatus;
enum {
    COKER_OSQP_OK = 0,
    COKER_OSQP_INVALID_ARGUMENT = -1,
    COKER_OSQP_INVALID_SHAPE = -2,
    COKER_OSQP_LAYOUT_MISMATCH = -3,
    COKER_OSQP_INVALID_NUMERIC_UPDATE = -4,
    COKER_OSQP_NOT_BOUND = -5,
    COKER_OSQP_UNSUPPORTED = -6,
};

typedef int32_t CokerOsqpSolveStatus;
enum {
    COKER_OSQP_SOLVE_UNSOLVED = 0,
    COKER_OSQP_SOLVE_SOLVED = 1,
    COKER_OSQP_SOLVE_SOLVED_INACCURATE = 2,
    COKER_OSQP_SOLVE_PRIMAL_INFEASIBLE_INACCURATE = 3,
    COKER_OSQP_SOLVE_DUAL_INFEASIBLE_INACCURATE = 4,
    COKER_OSQP_SOLVE_MAX_ITER_REACHED = -2,
    COKER_OSQP_SOLVE_PRIMAL_INFEASIBLE = -3,
    COKER_OSQP_SOLVE_DUAL_INFEASIBLE = -4,
    COKER_OSQP_SOLVE_INTERRUPTED = -5,
    COKER_OSQP_SOLVE_TIME_LIMIT_REACHED = -6,
    COKER_OSQP_SOLVE_NON_CONVEX = -7,
};

typedef struct {
    const float *primal;
    int32_t primal_len;
    const float *dual;
    int32_t dual_len;
    CokerOsqpSolveStatus status;
    int32_t iterations;
    float primal_residual;
    float dual_residual;
} CokerOsqpSolution;

CokerOsqpStatus coker_osqp_layout_for_shape(const CokerOsqpProblemShape *shape,
                                             CokerOsqpLayout *layout);
CokerOsqpStatus coker_osqp_bind(const CokerOsqpProblemShape *shape,
                                 const CokerOsqpLayout *layout,
                                 const CokerOsqpBuffers *buffers,
                                 CokerOsqpInstance *instance);
CokerOsqpStatus coker_osqp_bind_plan(const CokerOsqpPlan *plan,
                                     CokerOsqpArena arena,
                                     CokerOsqpInstance *instance);
CokerOsqpStatus coker_osqp_update(CokerOsqpInstance *instance,
                                   const CokerOsqpNumericUpdate *update);
CokerOsqpStatus coker_osqp_solve(CokerOsqpInstance *instance,
                                  CokerOsqpSolveStatus *solve_status);
CokerOsqpStatus coker_osqp_solution(const CokerOsqpInstance *instance,
                                     CokerOsqpSolution *solution);

#ifdef __cplusplus
}
#endif

#endif
#endif
