#include "coker_osqp_abi.h"
#include "lin_sys/direct/qdldl/qdldl_interface.h"
#include "auxil.h"
#include "kkt.h"
#include "osqp.h"

#include <stddef.h>
#include <limits.h>
#include <stdint.h>
#include <string.h>
#include <float.h>


#define COKER_ALIGNOF(T) offsetof(struct { char byte; T value; }, value)
#define COKER_STATIC_ASSERT(name, condition) \
    typedef char coker_static_assert_##name[(condition) ? 1 : -1]

#if !defined(EMBEDDED) || (EMBEDDED != 2)
#error "coker_osqp_abi.c requires EMBEDDED=2"
#endif
#ifdef DLONG
#error "coker_osqp_abi.c requires DLONG to be disabled"
#endif
#ifndef DFLOAT
#error "coker_osqp_abi.c requires DFLOAT"
#endif
#ifdef ENABLE_MKL_PARDISO
#error "coker_osqp_abi.c requires MKL to be disabled"
#endif
#ifdef PRINTING
#error "coker_osqp_abi.c requires printing to be disabled"
#endif
#ifdef PROFILING
#error "coker_osqp_abi.c requires profiling to be disabled"
#endif

COKER_STATIC_ASSERT(pointer_width, sizeof(void *) == 4);
COKER_STATIC_ASSERT(c_float_width, sizeof(c_float) == 4);
COKER_STATIC_ASSERT(c_int_width, sizeof(c_int) == 4);
COKER_STATIC_ASSERT(abi_float_width, sizeof(float) == sizeof(c_float));
COKER_STATIC_ASSERT(abi_index_width, sizeof(int32_t) == sizeof(c_int));
COKER_STATIC_ASSERT(abi_status_width, sizeof(CokerOsqpStatus) == sizeof(c_int));
COKER_STATIC_ASSERT(abi_solve_status_width,
                    sizeof(CokerOsqpSolveStatus) == sizeof(c_int));
COKER_STATIC_ASSERT(csc_align, COKER_ALIGNOF(csc) == 4);
COKER_STATIC_ASSERT(csc_column_pointer_offset, offsetof(csc, p) == 3 * sizeof(c_int));
COKER_STATIC_ASSERT(csc_row_index_offset,
                    offsetof(csc, i) == offsetof(csc, p) + sizeof(c_int *));
COKER_STATIC_ASSERT(csc_value_offset,
                    offsetof(csc, x) == offsetof(csc, i) + sizeof(c_int *));
COKER_STATIC_ASSERT(osqp_data_align, COKER_ALIGNOF(OSQPData) == 4);
COKER_STATIC_ASSERT(osqp_scaling_align, COKER_ALIGNOF(OSQPScaling) == 4);
COKER_STATIC_ASSERT(osqp_solution_align, COKER_ALIGNOF(OSQPSolution) == 4);
COKER_STATIC_ASSERT(osqp_info_align, COKER_ALIGNOF(OSQPInfo) == 4);
COKER_STATIC_ASSERT(linsys_solver_align, COKER_ALIGNOF(LinSysSolver) == 4);
COKER_STATIC_ASSERT(qdldl_solver_align, COKER_ALIGNOF(qdldl_solver) == 4);
COKER_STATIC_ASSERT(osqp_workspace_align, COKER_ALIGNOF(OSQPWorkspace) == 4);

static int coker_osqp_checked_add(coker_osqp_size_t left,
                                  coker_osqp_size_t right,
                                  coker_osqp_size_t *sum) {
    if (left > ULONG_MAX - right) {
        return 0;
    }
    *sum = left + right;
    return 1;
}

static int coker_osqp_checked_multiply(coker_osqp_size_t left,
                                       coker_osqp_size_t right,
                                       coker_osqp_size_t *product) {
    if (left != 0 && right > ULONG_MAX / left) {
        return 0;
    }
    *product = left * right;
    return 1;
}

static int coker_osqp_layout_region(CokerOsqpLayoutRegion *region,
                                    coker_osqp_size_t count,
                                    size_t element_size,
                                    size_t alignment) {
    coker_osqp_size_t bytes;

    if (element_size > ULONG_MAX || alignment == 0 || alignment > ULONG_MAX ||
        !coker_osqp_checked_multiply(count,
                                     (coker_osqp_size_t)element_size,
                                     &bytes)) {
        return 0;
    }
    region->bytes = bytes;
    region->alignment = (coker_osqp_size_t)alignment;
    return 1;
}

static int coker_osqp_csc_view_is_locally_valid(const CokerOsqpCscView *view,
                                                c_int nrows,
                                                c_int ncols,
                                                int upper_triangular) {
    c_int col;

    if (!view || !view->col_ptr || view->nnz < 0) {
        return 0;
    }
    if (((uintptr_t)view->col_ptr % COKER_ALIGNOF(c_int)) != 0 ||
        (view->nnz != 0 &&
         ((uintptr_t)view->row_idx % COKER_ALIGNOF(c_int)) != 0)) {
        return 0;
    }
    if (view->nnz != 0 && !view->row_idx) {
        return 0;
    }
    if (view->col_ptr[0] != 0 || view->col_ptr[ncols] != view->nnz) {
        return 0;
    }

    for (col = 0; col < ncols; ++col) {
        c_int entry;
        const c_int start = view->col_ptr[col];
        const c_int end = view->col_ptr[col + 1];
        c_int previous_row = -1;

        if (start < 0 || end < start || end > view->nnz) {
            return 0;
        }
        for (entry = start; entry < end; ++entry) {
            const c_int row = view->row_idx[entry];
            if (row < 0 || row >= nrows || row <= previous_row) {
                return 0;
            }
            if (upper_triangular && row > col) {
                return 0;
            }
            previous_row = row;
        }
    }
    return 1;
}

static int coker_osqp_csc_is_bound_valid(const csc *matrix,
                                         c_int nrows,
                                         c_int ncols,
                                         int upper_triangular) {
    c_int col;

    if (!matrix || !matrix->p || matrix->nzmax < 0 ||
        (matrix->nzmax != 0 && (!matrix->i || !matrix->x)) ||
        matrix->p[0] != 0 || matrix->p[ncols] != matrix->nzmax) {
        return 0;
    }
    for (col = 0; col < ncols; ++col) {
        c_int entry;
        const c_int start = matrix->p[col];
        const c_int end = matrix->p[col + 1];
        c_int previous_row = -1;

        if (start < 0 || end < start || end > matrix->nzmax) {
            return 0;
        }
        for (entry = start; entry < end; ++entry) {
            const c_int row = matrix->i[entry];
            if (row < 0 || row >= nrows || row <= previous_row ||
                (upper_triangular && row > col)) {
                return 0;
            }
            previous_row = row;
        }
    }
    return 1;
}

static int coker_osqp_shape_is_locally_valid(const CokerOsqpProblemShape *shape) {
    if (!shape || shape->n <= 0 || shape->m < 0) {
        return 0;
    }
    return coker_osqp_csc_view_is_locally_valid(&shape->p, shape->n, shape->n, 1) &&
           coker_osqp_csc_view_is_locally_valid(&shape->a, shape->m, shape->n, 0);
}

static int coker_osqp_numeric_update_is_locally_valid(
    const CokerOsqpNumericUpdate *update) {
    if (!update || update->p_nnz < 0 || update->a_nnz < 0 || update->q_len < 0 ||
        update->l_len < 0 || update->u_len < 0) {
        return 0;
    }
    return update->p_x != NULL && update->a_x != NULL && update->q != NULL &&
           update->l != NULL && update->u != NULL;
}

static int coker_osqp_values_are_finite(const c_float *values, c_int count) {
    c_int index;

    for (index = 0; index < count; ++index) {
        if (!(values[index] >= -FLT_MAX && values[index] <= FLT_MAX)) {
            return 0;
        }
    }
    return 1;
}

static int coker_osqp_status_is_valid(c_int status) {
    return status == OSQP_UNSOLVED || status == OSQP_SOLVED ||
           status == OSQP_SOLVED_INACCURATE ||
           status == OSQP_PRIMAL_INFEASIBLE_INACCURATE ||
           status == OSQP_DUAL_INFEASIBLE_INACCURATE ||
           status == OSQP_MAX_ITER_REACHED ||
           status == OSQP_PRIMAL_INFEASIBLE ||
           status == OSQP_DUAL_INFEASIBLE || status == OSQP_SIGINT ||
           status == OSQP_NON_CVX;
}

static CokerOsqpSolveStatus coker_osqp_solve_status_from_native(
    c_int native_status) {
    switch (native_status) {
    case OSQP_SOLVED:
        return COKER_OSQP_SOLVE_SOLVED;
    case OSQP_SOLVED_INACCURATE:
        return COKER_OSQP_SOLVE_SOLVED_INACCURATE;
    case OSQP_PRIMAL_INFEASIBLE_INACCURATE:
        return COKER_OSQP_SOLVE_PRIMAL_INFEASIBLE_INACCURATE;
    case OSQP_DUAL_INFEASIBLE_INACCURATE:
        return COKER_OSQP_SOLVE_DUAL_INFEASIBLE_INACCURATE;
    case OSQP_MAX_ITER_REACHED:
        return COKER_OSQP_SOLVE_MAX_ITER_REACHED;
    case OSQP_PRIMAL_INFEASIBLE:
        return COKER_OSQP_SOLVE_PRIMAL_INFEASIBLE;
    case OSQP_DUAL_INFEASIBLE:
        return COKER_OSQP_SOLVE_DUAL_INFEASIBLE;
    case OSQP_SIGINT:
        return COKER_OSQP_SOLVE_INTERRUPTED;
#ifdef PROFILING
    case OSQP_TIME_LIMIT_REACHED:
        return COKER_OSQP_SOLVE_TIME_LIMIT_REACHED;
#endif
    case OSQP_NON_CVX:
        return COKER_OSQP_SOLVE_NON_CONVEX;
    case OSQP_UNSOLVED:
    default:
        return COKER_OSQP_SOLVE_UNSOLVED;
    }
}

static int coker_osqp_instance_is_bound(const CokerOsqpInstance *instance) {
    const csc *pdata;
    const csc *adata;
    const OSQPData *data;
    const OSQPSettings *settings;
    const OSQPInfo *info;
    const OSQPWorkspace *workspace;
    const qdldl_solver *qdldl;

    if (!instance || !instance->pdata || !instance->adata || !instance->data ||
        !instance->settings || instance->scaling || !instance->solution ||
        !instance->info || !instance->linsys_solver || !instance->qdldl ||
        !instance->workspace) {
        return 0;
    }

    pdata = (const csc *)instance->pdata;
    adata = (const csc *)instance->adata;
    data = (const OSQPData *)instance->data;
    settings = (const OSQPSettings *)instance->settings;
    info = (const OSQPInfo *)instance->info;
    workspace = (const OSQPWorkspace *)instance->workspace;
    qdldl = (const qdldl_solver *)instance->qdldl;

    if (data->n <= 0 || data->m < 0 || data->P != pdata || data->A != adata ||
        !data->q || !data->l || !data->u || pdata->m != data->n ||
        pdata->n != data->n || adata->m != data->m || adata->n != data->n ||
        pdata->nz != -1 || adata->nz != -1 || !pdata->p || !adata->p ||
        (pdata->nzmax != 0 && (!pdata->i || !pdata->x)) ||
        (adata->nzmax != 0 && (!adata->i || !adata->x))) {
        return 0;
    }

    if (!coker_osqp_csc_is_bound_valid(pdata, data->n, data->n, 1) ||
        !coker_osqp_csc_is_bound_valid(adata, data->m, data->n, 0)) {
        return 0;
    }

    if (settings->scaling != 0 || workspace->data != data ||
        workspace->settings != settings || workspace->scaling != NULL ||
        workspace->info != info || workspace->linsys_solver != instance->linsys_solver ||
        instance->linsys_solver != instance->qdldl || qdldl->n != data->n ||
        qdldl->m != data->m || !qdldl->KKT || !qdldl->L || !qdldl->D ||
        !qdldl->Dinv || !qdldl->P || !qdldl->bp || !qdldl->sol ||
        !qdldl->PtoKKT || !qdldl->AtoKKT || !qdldl->rhotoKKT ||
        !qdldl->rho_inv_vec || !qdldl->Pdiag_idx || !qdldl->etree ||
        !qdldl->Lnz || !qdldl->iwork || !qdldl->bwork || !qdldl->fwork ||
        !qdldl->update_matrices || !qdldl->update_rho_vec ||
        !coker_osqp_status_is_valid(info->status_val)) {
        return 0;
    }

    return 1;
}

static coker_osqp_size_t coker_osqp_p_diagonal_count(
    const CokerOsqpProblemShape *shape) {
    coker_osqp_size_t count = 0;
    c_int column;

    for (column = 0; column < shape->n; ++column) {
        c_int entry;
        for (entry = shape->p.col_ptr[column];
             entry < shape->p.col_ptr[column + 1];
             ++entry) {
            if (shape->p.row_idx[entry] == column) {
                ++count;
                break;
            }
        }
    }
    return count;
}

CokerOsqpStatus coker_osqp_layout_for_shape(const CokerOsqpProblemShape *shape,
                                             CokerOsqpLayout *layout) {
    CokerOsqpLayout result;
    coker_osqp_size_t n;
    coker_osqp_size_t m;
    coker_osqp_size_t p_nnz;
    coker_osqp_size_t a_nnz;
    coker_osqp_size_t p_diagonal;
    coker_osqp_size_t missing_p_diagonal;
    coker_osqp_size_t n_plus_m;
    coker_osqp_size_t n_plus_m_plus_one;
    coker_osqp_size_t n_plus_one;
    coker_osqp_size_t kkt_nnz;
    coker_osqp_size_t l_capacity;
    coker_osqp_size_t iwork_count;

    if (!layout) {
        return COKER_OSQP_INVALID_ARGUMENT;
    }
    memset(layout, 0, sizeof(*layout));
    if (!shape) {
        return COKER_OSQP_INVALID_ARGUMENT;
    }
    if (!coker_osqp_shape_is_locally_valid(shape)) {
        return COKER_OSQP_INVALID_SHAPE;
    }

    n = (coker_osqp_size_t)shape->n;
    m = (coker_osqp_size_t)shape->m;
    p_nnz = (coker_osqp_size_t)shape->p.nnz;
    a_nnz = (coker_osqp_size_t)shape->a.nnz;
    p_diagonal = coker_osqp_p_diagonal_count(shape);
    if (p_diagonal > n) {
        return COKER_OSQP_INVALID_SHAPE;
    }
    missing_p_diagonal = n - p_diagonal;

    if (!coker_osqp_checked_add(n, m, &n_plus_m) ||
        n_plus_m > (coker_osqp_size_t)INT_MAX ||
        !coker_osqp_checked_add(n, 1, &n_plus_one) ||
        !coker_osqp_checked_add(n_plus_m, 1, &n_plus_m_plus_one) ||
        !coker_osqp_checked_add(p_nnz, missing_p_diagonal, &kkt_nnz) ||
        !coker_osqp_checked_add(kkt_nnz, a_nnz, &kkt_nnz) ||
        !coker_osqp_checked_add(kkt_nnz, m, &kkt_nnz) ||
        kkt_nnz > (coker_osqp_size_t)INT_MAX ||
        !coker_osqp_checked_multiply(n_plus_m, 3, &iwork_count) ||
        iwork_count > (coker_osqp_size_t)INT_MAX) {
        return COKER_OSQP_INVALID_SHAPE;
    }

    /*
     * QDLDL's symbolic analysis determines the exact L capacity only after
     * consuming KKT scratch.  Reserve the checked dense strictly-lower
     * triangular bound N * (N - 1) / 2 instead, where N = n + m.
     */
    if (n_plus_m > 0) {
        coker_osqp_size_t l_left = n_plus_m;
        coker_osqp_size_t l_right = n_plus_m - 1;
        if ((l_left & 1u) == 0) {
            l_left /= 2;
        } else {
            l_right /= 2;
        }
        if (!coker_osqp_checked_multiply(l_left, l_right, &l_capacity) ||
            l_capacity > (coker_osqp_size_t)INT_MAX) {
            return COKER_OSQP_INVALID_SHAPE;
        }
    } else {
        l_capacity = 0;
    }

    memset(&result, 0, sizeof(result));
#define COKER_LAYOUT_REGION(member, count, type) \
    do { \
        if (!coker_osqp_layout_region(&result.member, \
                                      (count), \
                                      sizeof(type), \
                                      COKER_ALIGNOF(type))) { \
            return COKER_OSQP_INVALID_SHAPE; \
        } \
    } while (0)

    COKER_LAYOUT_REGION(pdata_p, n_plus_one, c_int);
    COKER_LAYOUT_REGION(pdata_i, p_nnz, c_int);
    COKER_LAYOUT_REGION(pdata_x, p_nnz, c_float);
    COKER_LAYOUT_REGION(pdata, 1, csc);
    COKER_LAYOUT_REGION(adata_p, n_plus_one, c_int);
    COKER_LAYOUT_REGION(adata_i, a_nnz, c_int);
    COKER_LAYOUT_REGION(adata_x, a_nnz, c_float);
    COKER_LAYOUT_REGION(adata, 1, csc);
    COKER_LAYOUT_REGION(qdata, n, c_float);
    COKER_LAYOUT_REGION(ldata, m, c_float);
    COKER_LAYOUT_REGION(udata, m, c_float);
    COKER_LAYOUT_REGION(data, 1, OSQPData);
    COKER_LAYOUT_REGION(settings, 1, OSQPSettings);
    COKER_LAYOUT_REGION(scaling, 0, OSQPScaling);
    COKER_LAYOUT_REGION(xsolution, n, c_float);
    COKER_LAYOUT_REGION(ysolution, m, c_float);
    COKER_LAYOUT_REGION(solution, 1, OSQPSolution);
    COKER_LAYOUT_REGION(info, 1, OSQPInfo);
    COKER_LAYOUT_REGION(qdldl_L, 1, csc);
    COKER_LAYOUT_REGION(qdldl_L_p, n_plus_m_plus_one, QDLDL_int);
    COKER_LAYOUT_REGION(qdldl_L_i, l_capacity, QDLDL_int);
    COKER_LAYOUT_REGION(qdldl_L_x, l_capacity, QDLDL_float);
    COKER_LAYOUT_REGION(qdldl_KKT, 1, csc);
    COKER_LAYOUT_REGION(qdldl_KKT_p, n_plus_m_plus_one, c_int);
    COKER_LAYOUT_REGION(qdldl_KKT_i, kkt_nnz, c_int);
    COKER_LAYOUT_REGION(qdldl_KKT_x, kkt_nnz, c_float);
    COKER_LAYOUT_REGION(qdldl, 1, qdldl_solver);
    COKER_LAYOUT_REGION(qdldl_Dinv, n_plus_m, QDLDL_float);
    COKER_LAYOUT_REGION(qdldl_P, n_plus_m, QDLDL_int);
    COKER_LAYOUT_REGION(qdldl_bp, n_plus_m, QDLDL_float);
    COKER_LAYOUT_REGION(qdldl_sol, n_plus_m, QDLDL_float);
    COKER_LAYOUT_REGION(qdldl_rho_inv_vec, m, c_float);
    COKER_LAYOUT_REGION(qdldl_Pdiag_idx, p_diagonal, c_int);
    COKER_LAYOUT_REGION(qdldl_PtoKKT, p_nnz, c_int);
    COKER_LAYOUT_REGION(qdldl_AtoKKT, a_nnz, c_int);
    COKER_LAYOUT_REGION(qdldl_rhotoKKT, m, c_int);
    COKER_LAYOUT_REGION(qdldl_D, n_plus_m, QDLDL_float);
    COKER_LAYOUT_REGION(qdldl_etree, n_plus_m, QDLDL_int);
    COKER_LAYOUT_REGION(qdldl_Lnz, n_plus_m, QDLDL_int);
    COKER_LAYOUT_REGION(qdldl_iwork, iwork_count, QDLDL_int);
    COKER_LAYOUT_REGION(qdldl_bwork, n_plus_m, QDLDL_bool);
    COKER_LAYOUT_REGION(qdldl_fwork, n_plus_m, QDLDL_float);
    COKER_LAYOUT_REGION(work_rho_vec, m, c_float);
    COKER_LAYOUT_REGION(work_rho_inv_vec, m, c_float);
    COKER_LAYOUT_REGION(work_constr_type, m, c_int);
    COKER_LAYOUT_REGION(work_x, n, c_float);
    COKER_LAYOUT_REGION(work_y, m, c_float);
    COKER_LAYOUT_REGION(work_z, m, c_float);
    COKER_LAYOUT_REGION(work_xz_tilde, n_plus_m, c_float);
    COKER_LAYOUT_REGION(work_x_prev, n, c_float);
    COKER_LAYOUT_REGION(work_z_prev, m, c_float);
    COKER_LAYOUT_REGION(work_Ax, m, c_float);
    COKER_LAYOUT_REGION(work_Px, n, c_float);
    COKER_LAYOUT_REGION(work_Aty, n, c_float);
    COKER_LAYOUT_REGION(work_delta_y, m, c_float);
    COKER_LAYOUT_REGION(work_Atdelta_y, n, c_float);
    COKER_LAYOUT_REGION(work_delta_x, n, c_float);
    COKER_LAYOUT_REGION(work_Pdelta_x, n, c_float);
    COKER_LAYOUT_REGION(work_Adelta_x, m, c_float);
    COKER_LAYOUT_REGION(work_D_temp, 0, c_float);
    COKER_LAYOUT_REGION(work_D_temp_A, 0, c_float);
    COKER_LAYOUT_REGION(work_E_temp, 0, c_float);
    COKER_LAYOUT_REGION(workspace, 1, OSQPWorkspace);

#undef COKER_LAYOUT_REGION
    *layout = result;
    return COKER_OSQP_OK;
}

#define COKER_OSQP_FOR_EACH_REGION(X) \
    X(pdata_p) X(pdata_i) X(pdata_x) X(pdata) \
    X(adata_p) X(adata_i) X(adata_x) X(adata) \
    X(qdata) X(ldata) X(udata) X(data) X(settings) X(scaling) \
    X(xsolution) X(ysolution) X(solution) X(info) \
    X(qdldl_L) X(qdldl_L_p) X(qdldl_L_i) X(qdldl_L_x) \
    X(qdldl_KKT) X(qdldl_KKT_p) X(qdldl_KKT_i) X(qdldl_KKT_x) \
    X(qdldl) X(qdldl_Dinv) X(qdldl_P) X(qdldl_bp) X(qdldl_sol) \
    X(qdldl_rho_inv_vec) X(qdldl_Pdiag_idx) X(qdldl_PtoKKT) \
    X(qdldl_AtoKKT) X(qdldl_rhotoKKT) X(qdldl_D) X(qdldl_etree) \
    X(qdldl_Lnz) X(qdldl_iwork) X(qdldl_bwork) X(qdldl_fwork) \
    X(work_rho_vec) X(work_rho_inv_vec) X(work_constr_type) \
    X(work_x) X(work_y) X(work_z) X(work_xz_tilde) X(work_x_prev) \
    X(work_z_prev) X(work_Ax) X(work_Px) X(work_Aty) X(work_delta_y) \
    X(work_Atdelta_y) X(work_delta_x) X(work_Pdelta_x) X(work_Adelta_x) \
    X(work_D_temp) X(work_D_temp_A) X(work_E_temp) X(workspace)

static int coker_osqp_layout_matches(const CokerOsqpLayout *actual,
                                     const CokerOsqpLayout *expected) {
#define COKER_OSQP_LAYOUT_MATCH(member) \
    if (actual->member.bytes != expected->member.bytes || \
        actual->member.alignment != expected->member.alignment) return 0;
    COKER_OSQP_FOR_EACH_REGION(COKER_OSQP_LAYOUT_MATCH)
#undef COKER_OSQP_LAYOUT_MATCH
    return 1;
}

static int coker_osqp_buffer_region_is_valid(
    const CokerOsqpBufferRegion *region,
    const CokerOsqpLayoutRegion *expected) {
    const uintptr_t address = (uintptr_t)region->ptr;

    return region->ptr != NULL && region->bytes == expected->bytes &&
           region->alignment == expected->alignment &&
           address % (uintptr_t)expected->alignment == 0;
}

static int coker_osqp_buffers_are_valid(const CokerOsqpBuffers *buffers,
                                        const CokerOsqpLayout *layout) {
    const CokerOsqpBufferRegion *regions[] = {
#define COKER_OSQP_REGION_POINTER(member) &buffers->member,
        COKER_OSQP_FOR_EACH_REGION(COKER_OSQP_REGION_POINTER)
#undef COKER_OSQP_REGION_POINTER
    };
    const coker_osqp_size_t count =
        (coker_osqp_size_t)(sizeof(regions) / sizeof(regions[0]));
    coker_osqp_size_t left;

#define COKER_OSQP_REGION_MATCH(member) \
    if (!coker_osqp_buffer_region_is_valid(&buffers->member, &layout->member)) return 0;
    COKER_OSQP_FOR_EACH_REGION(COKER_OSQP_REGION_MATCH)
#undef COKER_OSQP_REGION_MATCH

    for (left = 0; left < count; ++left) {
        uintptr_t left_begin;
        uintptr_t left_end;
        coker_osqp_size_t right;

        if (regions[left]->bytes == 0) {
            continue;
        }
        left_begin = (uintptr_t)regions[left]->ptr;
        if ((uintptr_t)regions[left]->bytes > UINTPTR_MAX - left_begin) {
            return 0;
        }
        left_end = left_begin + (uintptr_t)regions[left]->bytes;
        for (right = left + 1; right < count; ++right) {
            uintptr_t right_begin;
            uintptr_t right_end;

            if (regions[right]->bytes == 0) {
                continue;
            }
            right_begin = (uintptr_t)regions[right]->ptr;
            if ((uintptr_t)regions[right]->bytes > UINTPTR_MAX - right_begin) {
                return 0;
            }
            right_end = right_begin + (uintptr_t)regions[right]->bytes;
            if (left_begin < right_end && right_begin < left_end) {
                return 0;
            }
        }
    }
    return 1;
}

static void coker_osqp_zero_buffers(const CokerOsqpBuffers *buffers,
                                    const CokerOsqpLayout *layout) {
#define COKER_OSQP_ZERO_REGION(member) \
    memset(buffers->member.ptr, 0, (size_t)layout->member.bytes);
    COKER_OSQP_FOR_EACH_REGION(COKER_OSQP_ZERO_REGION)
#undef COKER_OSQP_ZERO_REGION
}


#define COKER_OSQP_FOR_EACH_ARENA_REGION(X) \
    X(pdata_x) X(pdata) X(adata_x) X(adata) X(qdata) X(ldata) X(udata) \
    X(data) X(settings) X(xsolution) X(ysolution) X(solution) X(info) \
    X(qdldl_L_x) X(qdldl_L) X(qdldl_KKT_x) X(qdldl_KKT) X(qdldl) \
    X(qdldl_Dinv) X(qdldl_bp) X(qdldl_sol) X(qdldl_rho_inv_vec) \
    X(qdldl_D) X(qdldl_iwork) X(qdldl_bwork) X(qdldl_fwork) \
    X(work_rho_vec) X(work_rho_inv_vec) X(work_constr_type) X(work_x) \
    X(work_y) X(work_z) X(work_xz_tilde) X(work_x_prev) X(work_z_prev) \
    X(work_Ax) X(work_Px) X(work_Aty) X(work_delta_y) X(work_Atdelta_y) \
    X(work_delta_x) X(work_Pdelta_x) X(work_Adelta_x) X(workspace)

static int coker_osqp_index_view_is_locally_valid(
    const CokerOsqpIndexView *view) {
    return view && view->len >= 0 &&
           (view->len == 0 ||
            (view->ptr != NULL &&
             ((uintptr_t)view->ptr % COKER_ALIGNOF(c_int)) == 0));
}

static int coker_osqp_index_view_values_are_in_range(
    const CokerOsqpIndexView *view,
    c_int lower_bound,
    c_int upper_bound_exclusive) {
    c_int index;

    if (!coker_osqp_index_view_is_locally_valid(view) ||
        lower_bound > upper_bound_exclusive) {
        return 0;
    }
    for (index = 0; index < view->len; ++index) {
        const c_int value = view->ptr[index];
        if (value < lower_bound || value >= upper_bound_exclusive) {
            return 0;
        }
    }
    return 1;
}

static int coker_osqp_permutation_is_valid(const CokerOsqpIndexView *view,
                                           c_int len) {
    c_int left;

    if (!coker_osqp_index_view_values_are_in_range(view, 0, len) ||
        view->len != len) {
        return 0;
    }
    for (left = 0; left < len; ++left) {
        c_int right;
        for (right = left + 1; right < len; ++right) {
            if (view->ptr[left] == view->ptr[right]) {
                return 0;
            }
        }
    }
    return 1;
}

static coker_osqp_size_t coker_osqp_csc_diagonal_count(
    const CokerOsqpCscView *view,
    c_int n) {
    coker_osqp_size_t count = 0;
    c_int column;

    for (column = 0; column < n; ++column) {
        c_int entry;
        for (entry = view->col_ptr[column]; entry < view->col_ptr[column + 1];
             ++entry) {
            if (view->row_idx[entry] == column) {
                ++count;
                break;
            }
        }
    }
    return count;
}

static int coker_osqp_qdldl_l_is_locally_valid(const CokerOsqpCscView *view,
                                               c_int n) {
    c_int col;

    if (!view || !view->col_ptr || view->nnz < 0) {
        return 0;
    }
    if (((uintptr_t)view->col_ptr % COKER_ALIGNOF(c_int)) != 0 ||
        (view->nnz != 0 &&
         ((uintptr_t)view->row_idx % COKER_ALIGNOF(c_int)) != 0)) {
        return 0;
    }
    if (view->nnz != 0 && !view->row_idx) {
        return 0;
    }
    if (view->col_ptr[0] != 0 || view->col_ptr[n] != view->nnz) {
        return 0;
    }
    for (col = 0; col < n; ++col) {
        c_int entry;
        const c_int start = view->col_ptr[col];
        const c_int end = view->col_ptr[col + 1];
        c_int previous_row = col;

        if (start < 0 || end < start || end > view->nnz) {
            return 0;
        }
        for (entry = start; entry < end; ++entry) {
            const c_int row = view->row_idx[entry];
            if (row <= previous_row || row >= n) {
                return 0;
            }
            previous_row = row;
        }
    }
    return 1;
}

static int coker_osqp_p_diagonal_index_view_is_valid(
    const CokerOsqpCscView *p,
    c_int n,
    const CokerOsqpIndexView *diag) {
    c_int column;
    c_int diag_index = 0;

    if (!coker_osqp_index_view_is_locally_valid(diag)) {
        return 0;
    }
    for (column = 0; column < n; ++column) {
        c_int entry;
        for (entry = p->col_ptr[column]; entry < p->col_ptr[column + 1];
             ++entry) {
            if (p->row_idx[entry] == column) {
                if (diag_index >= diag->len || diag->ptr[diag_index] != entry) {
                    return 0;
                }
                ++diag_index;
                break;
            }
        }
    }
    return diag_index == diag->len;
}

static int coker_osqp_settings_are_locally_valid(
    const CokerOsqpSettings *settings) {
    if (!settings) {
        return 0;
    }
    if (!(settings->rho >= -FLT_MAX && settings->rho <= FLT_MAX) ||
        !(settings->sigma >= -FLT_MAX && settings->sigma <= FLT_MAX) ||
        !(settings->adaptive_rho_tolerance >= -FLT_MAX &&
          settings->adaptive_rho_tolerance <= FLT_MAX) ||
        !(settings->eps_abs >= -FLT_MAX && settings->eps_abs <= FLT_MAX) ||
        !(settings->eps_rel >= -FLT_MAX && settings->eps_rel <= FLT_MAX) ||
        !(settings->eps_prim_inf >= -FLT_MAX &&
          settings->eps_prim_inf <= FLT_MAX) ||
        !(settings->eps_dual_inf >= -FLT_MAX &&
          settings->eps_dual_inf <= FLT_MAX) ||
        !(settings->alpha >= -FLT_MAX && settings->alpha <= FLT_MAX)) {
        return 0;
    }
    if (settings->scaling != 0 || settings->rho <= 0.0f ||
        settings->sigma <= 0.0f || settings->adaptive_rho < 0 ||
        settings->adaptive_rho > 1 || settings->adaptive_rho_interval < 0 ||
        settings->adaptive_rho_tolerance < 1.0f || settings->max_iter <= 0 ||
        settings->eps_abs < 0.0f || settings->eps_rel < 0.0f ||
        (settings->eps_abs == 0.0f && settings->eps_rel == 0.0f) ||
        settings->eps_prim_inf <= 0.0f || settings->eps_dual_inf <= 0.0f ||
        settings->alpha <= 0.0f || settings->alpha >= 2.0f ||
        settings->linsys_solver != (uint32_t)QDLDL_SOLVER ||
        (settings->scaled_termination != 0 &&
         settings->scaled_termination != 1) ||
        settings->check_termination < 0 ||
        (settings->warm_start != 0 && settings->warm_start != 1)) {
        return 0;
    }
    return 1;
}
static int coker_osqp_arena_layout_region(CokerOsqpArenaRegion *region,
                                          coker_osqp_size_t count,
                                          size_t element_size,
                                          size_t alignment) {
    coker_osqp_size_t bytes;

    if (!region || element_size > ULONG_MAX || alignment == 0 ||
        alignment > ULONG_MAX ||
        !coker_osqp_checked_multiply(count,
                                     (coker_osqp_size_t)element_size,
                                     &bytes)) {
        return 0;
    }
    region->offset = 0;
    region->bytes = bytes;
    region->alignment = (coker_osqp_size_t)alignment;
    return 1;
}


static int coker_osqp_expected_arena_layout_for_plan(
    const CokerOsqpPlan *plan,
    CokerOsqpArenaLayout *layout) {
    CokerOsqpArenaLayout result;
    coker_osqp_size_t n;
    coker_osqp_size_t m;
    coker_osqp_size_t n_plus_m;
    coker_osqp_size_t p_nnz;
    coker_osqp_size_t a_nnz;
    coker_osqp_size_t kkt_nnz;
    coker_osqp_size_t l_nnz;
    coker_osqp_size_t iwork_count;

    if (!plan || !layout || plan->n <= 0 || plan->m < 0 || plan->n_plus_m <= 0 ||
        plan->p.nnz < 0 || plan->a.nnz < 0 || plan->kkt.nnz < 0 ||
        plan->qdldl_l.nnz < 0) {
        return 0;
    }

    n = (coker_osqp_size_t)plan->n;
    m = (coker_osqp_size_t)plan->m;
    n_plus_m = (coker_osqp_size_t)plan->n_plus_m;
    p_nnz = (coker_osqp_size_t)plan->p.nnz;
    a_nnz = (coker_osqp_size_t)plan->a.nnz;
    kkt_nnz = (coker_osqp_size_t)plan->kkt.nnz;
    l_nnz = (coker_osqp_size_t)plan->qdldl_l.nnz;
    if (!coker_osqp_checked_multiply(n_plus_m, 3u, &iwork_count) ||
        iwork_count > (coker_osqp_size_t)INT_MAX) {
        return 0;
    }

    memset(&result, 0, sizeof(result));
    result.alignment = 1;
#define COKER_ARENA_LAYOUT_REGION(member, count, type) \
    do { \
        if (!coker_osqp_arena_layout_region(&result.member, \
                                            (count), \
                                            sizeof(type), \
                                            COKER_ALIGNOF(type))) { \
            return 0; \
        } \
        if (result.member.alignment > result.alignment) { \
            result.alignment = result.member.alignment; \
        } \
    } while (0)

    COKER_ARENA_LAYOUT_REGION(pdata_x, p_nnz, c_float);
    COKER_ARENA_LAYOUT_REGION(pdata, 1, csc);
    COKER_ARENA_LAYOUT_REGION(adata_x, a_nnz, c_float);
    COKER_ARENA_LAYOUT_REGION(adata, 1, csc);
    COKER_ARENA_LAYOUT_REGION(qdata, n, c_float);
    COKER_ARENA_LAYOUT_REGION(ldata, m, c_float);
    COKER_ARENA_LAYOUT_REGION(udata, m, c_float);
    COKER_ARENA_LAYOUT_REGION(data, 1, OSQPData);
    COKER_ARENA_LAYOUT_REGION(settings, 1, OSQPSettings);
    COKER_ARENA_LAYOUT_REGION(xsolution, n, c_float);
    COKER_ARENA_LAYOUT_REGION(ysolution, m, c_float);
    COKER_ARENA_LAYOUT_REGION(solution, 1, OSQPSolution);
    COKER_ARENA_LAYOUT_REGION(info, 1, OSQPInfo);
    COKER_ARENA_LAYOUT_REGION(qdldl_L_x, l_nnz, QDLDL_float);
    COKER_ARENA_LAYOUT_REGION(qdldl_L, 1, csc);
    COKER_ARENA_LAYOUT_REGION(qdldl_KKT_x, kkt_nnz, c_float);
    COKER_ARENA_LAYOUT_REGION(qdldl_KKT, 1, csc);
    COKER_ARENA_LAYOUT_REGION(qdldl, 1, qdldl_solver);
    COKER_ARENA_LAYOUT_REGION(qdldl_Dinv, n_plus_m, QDLDL_float);
    COKER_ARENA_LAYOUT_REGION(qdldl_bp, n_plus_m, QDLDL_float);
    COKER_ARENA_LAYOUT_REGION(qdldl_sol, n_plus_m, QDLDL_float);
    COKER_ARENA_LAYOUT_REGION(qdldl_rho_inv_vec, m, c_float);
    COKER_ARENA_LAYOUT_REGION(qdldl_D, n_plus_m, QDLDL_float);
    COKER_ARENA_LAYOUT_REGION(qdldl_iwork, iwork_count, QDLDL_int);
    COKER_ARENA_LAYOUT_REGION(qdldl_bwork, n_plus_m, QDLDL_bool);
    COKER_ARENA_LAYOUT_REGION(qdldl_fwork, n_plus_m, QDLDL_float);
    COKER_ARENA_LAYOUT_REGION(work_rho_vec, m, c_float);
    COKER_ARENA_LAYOUT_REGION(work_rho_inv_vec, m, c_float);
    COKER_ARENA_LAYOUT_REGION(work_constr_type, m, c_int);
    COKER_ARENA_LAYOUT_REGION(work_x, n, c_float);
    COKER_ARENA_LAYOUT_REGION(work_y, m, c_float);
    COKER_ARENA_LAYOUT_REGION(work_z, m, c_float);
    COKER_ARENA_LAYOUT_REGION(work_xz_tilde, n_plus_m, c_float);
    COKER_ARENA_LAYOUT_REGION(work_x_prev, n, c_float);
    COKER_ARENA_LAYOUT_REGION(work_z_prev, m, c_float);
    COKER_ARENA_LAYOUT_REGION(work_Ax, m, c_float);
    COKER_ARENA_LAYOUT_REGION(work_Px, n, c_float);
    COKER_ARENA_LAYOUT_REGION(work_Aty, n, c_float);
    COKER_ARENA_LAYOUT_REGION(work_delta_y, m, c_float);
    COKER_ARENA_LAYOUT_REGION(work_Atdelta_y, n, c_float);
    COKER_ARENA_LAYOUT_REGION(work_delta_x, n, c_float);
    COKER_ARENA_LAYOUT_REGION(work_Pdelta_x, n, c_float);
    COKER_ARENA_LAYOUT_REGION(work_Adelta_x, m, c_float);
    COKER_ARENA_LAYOUT_REGION(workspace, 1, OSQPWorkspace);

#undef COKER_ARENA_LAYOUT_REGION
    *layout = result;
    return 1;
}

static int coker_osqp_plan_metadata_is_locally_valid(
    const CokerOsqpPlan *plan,
    CokerOsqpArenaLayout *expected_layout) {
    coker_osqp_size_t expected_n_plus_m;
    coker_osqp_size_t p_diagonal_count;
    coker_osqp_size_t expected_kkt_nnz;
    c_int index;

    if (!plan || !expected_layout || plan->n <= 0 || plan->m < 0 ||
        !coker_osqp_checked_add((coker_osqp_size_t)plan->n,
                                (coker_osqp_size_t)plan->m,
                                &expected_n_plus_m) ||
        expected_n_plus_m > (coker_osqp_size_t)INT_MAX ||
        plan->n_plus_m != (c_int)expected_n_plus_m ||
        !coker_osqp_csc_view_is_locally_valid(&plan->p, plan->n, plan->n, 1) ||
        !coker_osqp_csc_view_is_locally_valid(&plan->a, plan->m, plan->n, 0) ||
        !coker_osqp_csc_view_is_locally_valid(&plan->kkt,
                                              plan->n_plus_m,
                                              plan->n_plus_m,
                                              1) ||
        !coker_osqp_qdldl_l_is_locally_valid(&plan->qdldl_l,
                                             plan->n_plus_m) ||
        !coker_osqp_settings_are_locally_valid(&plan->settings)) {
        return 0;
    }

    p_diagonal_count = coker_osqp_csc_diagonal_count(&plan->p, plan->n);
    if (p_diagonal_count > (coker_osqp_size_t)plan->n ||
        !coker_osqp_checked_add((coker_osqp_size_t)plan->p.nnz,
                                (coker_osqp_size_t)plan->n - p_diagonal_count,
                                &expected_kkt_nnz) ||
        !coker_osqp_checked_add(expected_kkt_nnz,
                                (coker_osqp_size_t)plan->a.nnz,
                                &expected_kkt_nnz) ||
        !coker_osqp_checked_add(expected_kkt_nnz,
                                (coker_osqp_size_t)plan->m,
                                &expected_kkt_nnz) ||
        expected_kkt_nnz > (coker_osqp_size_t)INT_MAX ||
        plan->kkt.nnz != (c_int)expected_kkt_nnz ||
        !coker_osqp_index_view_values_are_in_range(&plan->p_to_kkt,
                                                   0,
                                                   plan->kkt.nnz) ||
        plan->p_to_kkt.len != plan->p.nnz ||
        !coker_osqp_index_view_values_are_in_range(&plan->a_to_kkt,
                                                   0,
                                                   plan->kkt.nnz) ||
        plan->a_to_kkt.len != plan->a.nnz ||
        !coker_osqp_index_view_values_are_in_range(&plan->rho_to_kkt,
                                                   0,
                                                   plan->kkt.nnz) ||
        plan->rho_to_kkt.len != plan->m ||
        !coker_osqp_p_diagonal_index_view_is_valid(&plan->p,
                                                   plan->n,
                                                   &plan->p_diagonal_idx) ||
        plan->p_diagonal_idx.len != (c_int)p_diagonal_count ||
        !coker_osqp_permutation_is_valid(&plan->permutation,
                                         plan->n_plus_m) ||
        !coker_osqp_index_view_is_locally_valid(&plan->qdldl_etree) ||
        !coker_osqp_index_view_is_locally_valid(&plan->qdldl_lnz) ||
        plan->qdldl_etree.len != plan->n_plus_m ||
        plan->qdldl_lnz.len != plan->n_plus_m) {
        return 0;
    }
    for (index = 0; index < plan->qdldl_etree.len; ++index) {
        if (plan->qdldl_etree.ptr[index] < -1 ||
            plan->qdldl_etree.ptr[index] >= plan->n_plus_m ||
            plan->qdldl_lnz.ptr[index] < 0) {
            return 0;
        }
    }

    return coker_osqp_expected_arena_layout_for_plan(plan, expected_layout);
}

static int coker_osqp_arena_layout_region_is_valid(
    const CokerOsqpArenaRegion *region,
    coker_osqp_size_t total_bytes) {
    coker_osqp_size_t end;

    if (!region || region->alignment == 0 ||
        region->offset % region->alignment != 0) {
        return 0;
    }
    if (region->offset > total_bytes) {
        return 0;
    }
    if (!coker_osqp_checked_add(region->offset, region->bytes, &end)) {
        return 0;
    }
    return end <= total_bytes;
}

static int coker_osqp_arena_layout_is_locally_valid(
    const CokerOsqpArenaLayout *layout,
    const CokerOsqpArenaLayout *expected) {
    coker_osqp_size_t left;

    if (!layout || !expected || layout->alignment < expected->alignment ||
        layout->bytes == 0) {
        return 0;
    }
    {
        const CokerOsqpArenaRegion *regions[] = {
#define COKER_OSQP_ARENA_REGION_POINTER(member) &layout->member,
            COKER_OSQP_FOR_EACH_ARENA_REGION(COKER_OSQP_ARENA_REGION_POINTER)
#undef COKER_OSQP_ARENA_REGION_POINTER
        };
        const coker_osqp_size_t count =
            (coker_osqp_size_t)(sizeof(regions) / sizeof(regions[0]));

#define COKER_OSQP_EXPECTED_ARENA_REGION(member) \
    if (layout->member.bytes != expected->member.bytes || \
        layout->member.alignment != expected->member.alignment || \
        !coker_osqp_arena_layout_region_is_valid(&layout->member, layout->bytes)) return 0;
        COKER_OSQP_FOR_EACH_ARENA_REGION(COKER_OSQP_EXPECTED_ARENA_REGION)
#undef COKER_OSQP_EXPECTED_ARENA_REGION

        for (left = 0; left < count; ++left) {
            coker_osqp_size_t right;
            const coker_osqp_size_t left_begin = regions[left]->offset;
            const coker_osqp_size_t left_end = left_begin + regions[left]->bytes;

            if (regions[left]->bytes == 0) {
                continue;
            }
            for (right = left + 1; right < count; ++right) {
                const coker_osqp_size_t right_begin = regions[right]->offset;
                const coker_osqp_size_t right_end =
                    right_begin + regions[right]->bytes;

                if (regions[right]->bytes == 0) {
                    continue;
                }
                if (left_begin < right_end && right_begin < left_end) {
                    return 0;
                }
            }
        }
    }
    return 1;
}

static int coker_osqp_arena_is_valid(const CokerOsqpArena *arena,
                                     const CokerOsqpArenaLayout *layout) {
    const uintptr_t base =
        arena && arena->base ? (uintptr_t)arena->base : (uintptr_t)0;

    return arena && arena->base != NULL && layout != NULL &&
           arena->alignment >= layout->alignment && arena->bytes >= layout->bytes &&
           layout->alignment != 0 &&
           base % (uintptr_t)layout->alignment == 0;
}

static void *coker_osqp_arena_region_ptr(const CokerOsqpArena *arena,
                                         const CokerOsqpArenaRegion *region) {
    return (void *)((unsigned char *)arena->base + region->offset);
}

static void coker_osqp_zero_arena(const CokerOsqpArena *arena,
                                  const CokerOsqpArenaLayout *layout) {
#define COKER_OSQP_ZERO_ARENA_REGION(member) \
    memset(coker_osqp_arena_region_ptr(arena, &layout->member), \
           0, \
           (size_t)layout->member.bytes);
    COKER_OSQP_FOR_EACH_ARENA_REGION(COKER_OSQP_ZERO_ARENA_REGION)
#undef COKER_OSQP_ZERO_ARENA_REGION
}

static void coker_osqp_bind_csc_view(csc *matrix,
                                     c_int rows,
                                     c_int cols,
                                     const CokerOsqpCscView *view,
                                     c_float *x) {
    matrix->nzmax = view->nnz;
    matrix->m = rows;
    matrix->n = cols;
    matrix->p = (c_int *)view->col_ptr;
    matrix->i = (c_int *)view->row_idx;
    matrix->x = x;
    matrix->nz = -1;
}

static void coker_osqp_apply_settings(OSQPSettings *dst,
                                      const CokerOsqpSettings *src) {
    dst->rho = src->rho;
    dst->sigma = src->sigma;
    dst->scaling = src->scaling;
    dst->adaptive_rho = src->adaptive_rho;
    dst->adaptive_rho_interval = src->adaptive_rho_interval;
    dst->adaptive_rho_tolerance = src->adaptive_rho_tolerance;
    dst->max_iter = src->max_iter;
    dst->eps_abs = src->eps_abs;
    dst->eps_rel = src->eps_rel;
    dst->eps_prim_inf = src->eps_prim_inf;
    dst->eps_dual_inf = src->eps_dual_inf;
    dst->alpha = src->alpha;
    dst->linsys_solver = (enum linsys_solver_type)src->linsys_solver;
    dst->scaled_termination = src->scaled_termination;
    dst->check_termination = src->check_termination;
    dst->warm_start = src->warm_start;
}

static int coker_osqp_seed_kkt_sigma(csc *kkt, c_int n, c_float sigma) {
    c_int column;

    for (column = 0; column < n; ++column) {
        c_int entry;
        for (entry = kkt->p[column]; entry < kkt->p[column + 1]; ++entry) {
            if (kkt->i[entry] == column) {
                kkt->x[entry] = sigma;
                break;
            }
        }
        if (entry == kkt->p[column + 1]) {
            return 0;
        }
    }
    return 1;
}

CokerOsqpStatus coker_osqp_bind_plan(const CokerOsqpPlan *plan,
                                     CokerOsqpArena arena,
                                     CokerOsqpInstance *instance) {
    CokerOsqpArenaLayout expected_layout;
    csc *pdata;
    csc *adata;
    OSQPData *data;
    OSQPSettings *settings;
    OSQPSolution *solution;
    OSQPInfo *info;
    csc *linsys_l;
    csc *kkt;
    qdldl_solver *qdldl;
    OSQPWorkspace *workspace;
    c_int native_status;

    if (!instance) {
        return COKER_OSQP_INVALID_ARGUMENT;
    }
    memset(instance, 0, sizeof(*instance));
    if (!plan) {
        return COKER_OSQP_INVALID_ARGUMENT;
    }
    if (plan->abi_version != COKER_OSQP_PLAN_ABI_VERSION) {
        return COKER_OSQP_INVALID_ARGUMENT;
    }
    if (!coker_osqp_plan_metadata_is_locally_valid(plan, &expected_layout)) {
        return COKER_OSQP_INVALID_SHAPE;
    }
    if (!coker_osqp_arena_layout_is_locally_valid(&plan->arena_layout,
                                                  &expected_layout) ||
        !coker_osqp_arena_is_valid(&arena, &plan->arena_layout)) {
        return COKER_OSQP_LAYOUT_MISMATCH;
    }

    coker_osqp_zero_arena(&arena, &plan->arena_layout);

    pdata = (csc *)coker_osqp_arena_region_ptr(&arena,
                                               &plan->arena_layout.pdata);
    adata = (csc *)coker_osqp_arena_region_ptr(&arena,
                                               &plan->arena_layout.adata);
    data = (OSQPData *)coker_osqp_arena_region_ptr(&arena,
                                                   &plan->arena_layout.data);
    settings = (OSQPSettings *)coker_osqp_arena_region_ptr(
        &arena, &plan->arena_layout.settings);
    solution = (OSQPSolution *)coker_osqp_arena_region_ptr(
        &arena, &plan->arena_layout.solution);
    info = (OSQPInfo *)coker_osqp_arena_region_ptr(&arena,
                                                   &plan->arena_layout.info);
    linsys_l = (csc *)coker_osqp_arena_region_ptr(&arena,
                                                  &plan->arena_layout.qdldl_L);
    kkt = (csc *)coker_osqp_arena_region_ptr(&arena,
                                             &plan->arena_layout.qdldl_KKT);
    qdldl = (qdldl_solver *)coker_osqp_arena_region_ptr(
        &arena, &plan->arena_layout.qdldl);
    workspace = (OSQPWorkspace *)coker_osqp_arena_region_ptr(
        &arena, &plan->arena_layout.workspace);

    coker_osqp_bind_csc_view(
        pdata, plan->n, plan->n, &plan->p,
        (c_float *)coker_osqp_arena_region_ptr(&arena,
                                               &plan->arena_layout.pdata_x));
    coker_osqp_bind_csc_view(
        adata, plan->m, plan->n, &plan->a,
        (c_float *)coker_osqp_arena_region_ptr(&arena,
                                               &plan->arena_layout.adata_x));
    coker_osqp_bind_csc_view(
        linsys_l, plan->n_plus_m, plan->n_plus_m, &plan->qdldl_l,
        (c_float *)coker_osqp_arena_region_ptr(&arena,
                                               &plan->arena_layout.qdldl_L_x));
    coker_osqp_bind_csc_view(
        kkt, plan->n_plus_m, plan->n_plus_m, &plan->kkt,
        (c_float *)coker_osqp_arena_region_ptr(
            &arena, &plan->arena_layout.qdldl_KKT_x));

    data->n = plan->n;
    data->m = plan->m;
    data->P = pdata;
    data->A = adata;
    data->q = (c_float *)coker_osqp_arena_region_ptr(&arena,
                                                     &plan->arena_layout.qdata);
    data->l = (c_float *)coker_osqp_arena_region_ptr(&arena,
                                                     &plan->arena_layout.ldata);
    data->u = (c_float *)coker_osqp_arena_region_ptr(&arena,
                                                     &plan->arena_layout.udata);

    coker_osqp_apply_settings(settings, &plan->settings);

    solution->x = (c_float *)coker_osqp_arena_region_ptr(
        &arena, &plan->arena_layout.xsolution);
    solution->y = (c_float *)coker_osqp_arena_region_ptr(
        &arena, &plan->arena_layout.ysolution);

    reset_info(info);
    info->rho_estimate = settings->rho;

    qdldl->type = QDLDL_SOLVER;
    qdldl->solve = &solve_linsys_qdldl;
    qdldl->update_matrices = &update_linsys_solver_matrices_qdldl;
    qdldl->update_rho_vec = &update_linsys_solver_rho_vec_qdldl;
    qdldl->L = linsys_l;
    qdldl->Dinv = (QDLDL_float *)coker_osqp_arena_region_ptr(
        &arena, &plan->arena_layout.qdldl_Dinv);
    qdldl->P = (c_int *)plan->permutation.ptr;
    qdldl->bp = (QDLDL_float *)coker_osqp_arena_region_ptr(
        &arena, &plan->arena_layout.qdldl_bp);
    qdldl->sol = (QDLDL_float *)coker_osqp_arena_region_ptr(
        &arena, &plan->arena_layout.qdldl_sol);
    qdldl->rho_inv_vec = (c_float *)coker_osqp_arena_region_ptr(
        &arena, &plan->arena_layout.qdldl_rho_inv_vec);
    qdldl->sigma = settings->sigma;
    qdldl->n = plan->n;
    qdldl->m = plan->m;
    qdldl->Pdiag_idx = (c_int *)plan->p_diagonal_idx.ptr;
    qdldl->Pdiag_n = plan->p_diagonal_idx.len;
    qdldl->KKT = kkt;
    qdldl->PtoKKT = (c_int *)plan->p_to_kkt.ptr;
    qdldl->AtoKKT = (c_int *)plan->a_to_kkt.ptr;
    qdldl->rhotoKKT = (c_int *)plan->rho_to_kkt.ptr;
    qdldl->D = (QDLDL_float *)coker_osqp_arena_region_ptr(
        &arena, &plan->arena_layout.qdldl_D);
    qdldl->etree = (QDLDL_int *)plan->qdldl_etree.ptr;
    qdldl->Lnz = (QDLDL_int *)plan->qdldl_lnz.ptr;
    qdldl->iwork = (QDLDL_int *)coker_osqp_arena_region_ptr(
        &arena, &plan->arena_layout.qdldl_iwork);
    qdldl->bwork = (QDLDL_bool *)coker_osqp_arena_region_ptr(
        &arena, &plan->arena_layout.qdldl_bwork);
    qdldl->fwork = (QDLDL_float *)coker_osqp_arena_region_ptr(
        &arena, &plan->arena_layout.qdldl_fwork);

    workspace->data = data;
    workspace->linsys_solver = (LinSysSolver *)qdldl;
    workspace->rho_vec = (c_float *)coker_osqp_arena_region_ptr(
        &arena, &plan->arena_layout.work_rho_vec);
    workspace->rho_inv_vec = (c_float *)coker_osqp_arena_region_ptr(
        &arena, &plan->arena_layout.work_rho_inv_vec);
    workspace->constr_type = (c_int *)coker_osqp_arena_region_ptr(
        &arena, &plan->arena_layout.work_constr_type);
    workspace->x = (c_float *)coker_osqp_arena_region_ptr(
        &arena, &plan->arena_layout.work_x);
    workspace->y = (c_float *)coker_osqp_arena_region_ptr(
        &arena, &plan->arena_layout.work_y);
    workspace->z = (c_float *)coker_osqp_arena_region_ptr(
        &arena, &plan->arena_layout.work_z);
    workspace->xz_tilde = (c_float *)coker_osqp_arena_region_ptr(
        &arena, &plan->arena_layout.work_xz_tilde);
    workspace->x_prev = (c_float *)coker_osqp_arena_region_ptr(
        &arena, &plan->arena_layout.work_x_prev);
    workspace->z_prev = (c_float *)coker_osqp_arena_region_ptr(
        &arena, &plan->arena_layout.work_z_prev);
    workspace->Ax = (c_float *)coker_osqp_arena_region_ptr(
        &arena, &plan->arena_layout.work_Ax);
    workspace->Px = (c_float *)coker_osqp_arena_region_ptr(
        &arena, &plan->arena_layout.work_Px);
    workspace->Aty = (c_float *)coker_osqp_arena_region_ptr(
        &arena, &plan->arena_layout.work_Aty);
    workspace->delta_y = (c_float *)coker_osqp_arena_region_ptr(
        &arena, &plan->arena_layout.work_delta_y);
    workspace->Atdelta_y = (c_float *)coker_osqp_arena_region_ptr(
        &arena, &plan->arena_layout.work_Atdelta_y);
    workspace->delta_x = (c_float *)coker_osqp_arena_region_ptr(
        &arena, &plan->arena_layout.work_delta_x);
    workspace->Pdelta_x = (c_float *)coker_osqp_arena_region_ptr(
        &arena, &plan->arena_layout.work_Pdelta_x);
    workspace->Adelta_x = (c_float *)coker_osqp_arena_region_ptr(
        &arena, &plan->arena_layout.work_Adelta_x);
    workspace->D_temp = NULL;
    workspace->D_temp_A = NULL;
    workspace->E_temp = NULL;
    workspace->settings = settings;
    workspace->scaling = NULL;
    workspace->solution = solution;
    workspace->info = info;

    set_rho_vec(workspace);
    if (!coker_osqp_seed_kkt_sigma(kkt, plan->n, settings->sigma)) {
        return COKER_OSQP_INVALID_SHAPE;
    }
    if (plan->m != 0) {
        memmove(qdldl->rho_inv_vec,
                workspace->rho_inv_vec,
                (size_t)plan->arena_layout.qdldl_rho_inv_vec.bytes);
        update_KKT_param2(kkt, qdldl->rho_inv_vec, qdldl->rhotoKKT, plan->m);
    }
    native_status = qdldl->update_matrices(qdldl, data->P, data->A);
    if (native_status != 0) {
        memset(instance, 0, sizeof(*instance));
        return COKER_OSQP_INVALID_SHAPE;
    }

    instance->pdata = pdata;
    instance->adata = adata;
    instance->data = data;
    instance->settings = settings;
    instance->scaling = NULL;
    instance->solution = solution;
    instance->info = info;
    instance->linsys_solver = qdldl;
    instance->qdldl = qdldl;
    instance->workspace = workspace;
    return COKER_OSQP_OK;
}


CokerOsqpStatus coker_osqp_bind(const CokerOsqpProblemShape *shape,
                                 const CokerOsqpLayout *layout,
                                 const CokerOsqpBuffers *buffers,
                                 CokerOsqpInstance *instance) {
    CokerOsqpLayout expected_layout;
    csc *pdata;
    csc *adata;
    OSQPData *data;
    OSQPSettings *settings;
    OSQPSolution *solution;
    OSQPInfo *info;
    OSQPWorkspace *workspace;
    csc *kkt;
    csc *linsys_l;
    qdldl_solver *qdldl;
    c_int pdiag_n = 0;
    c_int native_status;

    if (!instance) {
        return COKER_OSQP_INVALID_ARGUMENT;
    }
    memset(instance, 0, sizeof(*instance));
    if (!shape || !layout || !buffers) {
        return COKER_OSQP_INVALID_ARGUMENT;
    }
    if (!coker_osqp_shape_is_locally_valid(shape)) {
        return COKER_OSQP_INVALID_SHAPE;
    }
    if (coker_osqp_layout_for_shape(shape, &expected_layout) != COKER_OSQP_OK ||
        !coker_osqp_layout_matches(layout, &expected_layout) ||
        !coker_osqp_buffers_are_valid(buffers, layout)) {
        return COKER_OSQP_LAYOUT_MISMATCH;
    }

    coker_osqp_zero_buffers(buffers, layout);

    pdata = (csc *)buffers->pdata.ptr;
    adata = (csc *)buffers->adata.ptr;
    data = (OSQPData *)buffers->data.ptr;
    settings = (OSQPSettings *)buffers->settings.ptr;
    solution = (OSQPSolution *)buffers->solution.ptr;
    info = (OSQPInfo *)buffers->info.ptr;
    workspace = (OSQPWorkspace *)buffers->workspace.ptr;
    kkt = (csc *)buffers->qdldl_KKT.ptr;
    linsys_l = (csc *)buffers->qdldl_L.ptr;
    qdldl = (qdldl_solver *)buffers->qdldl.ptr;

    memmove(buffers->pdata_p.ptr,
            shape->p.col_ptr,
            (size_t)layout->pdata_p.bytes);
    if (shape->p.nnz != 0) {
        memmove(buffers->pdata_i.ptr,
                shape->p.row_idx,
                (size_t)layout->pdata_i.bytes);
    }
    pdata->nzmax = shape->p.nnz;
    pdata->m = shape->n;
    pdata->n = shape->n;
    pdata->p = (c_int *)buffers->pdata_p.ptr;
    pdata->i = (c_int *)buffers->pdata_i.ptr;
    pdata->x = (c_float *)buffers->pdata_x.ptr;
    pdata->nz = -1;

    memmove(buffers->adata_p.ptr,
            shape->a.col_ptr,
            (size_t)layout->adata_p.bytes);
    if (shape->a.nnz != 0) {
        memmove(buffers->adata_i.ptr,
                shape->a.row_idx,
                (size_t)layout->adata_i.bytes);
    }
    adata->nzmax = shape->a.nnz;
    adata->m = shape->m;
    adata->n = shape->n;
    adata->p = (c_int *)buffers->adata_p.ptr;
    adata->i = (c_int *)buffers->adata_i.ptr;
    adata->x = (c_float *)buffers->adata_x.ptr;
    adata->nz = -1;

    data->n = shape->n;
    data->m = shape->m;
    data->P = pdata;
    data->A = adata;
    data->q = (c_float *)buffers->qdata.ptr;
    data->l = (c_float *)buffers->ldata.ptr;
    data->u = (c_float *)buffers->udata.ptr;

    settings->rho = RHO;
    settings->sigma = SIGMA;
    settings->scaling = 0;
    settings->adaptive_rho = 0;
    settings->adaptive_rho_interval = 0;
    settings->adaptive_rho_tolerance = ADAPTIVE_RHO_TOLERANCE;
    settings->max_iter = MAX_ITER;
    settings->eps_abs = EPS_ABS;
    settings->eps_rel = EPS_REL;
    settings->eps_prim_inf = EPS_PRIM_INF;
    settings->eps_dual_inf = EPS_DUAL_INF;
    settings->alpha = ALPHA;
    settings->linsys_solver = QDLDL_SOLVER;
    settings->scaled_termination = SCALED_TERMINATION;
    settings->check_termination = CHECK_TERMINATION;
    settings->warm_start = WARM_START;

    solution->x = (c_float *)buffers->xsolution.ptr;
    solution->y = (c_float *)buffers->ysolution.ptr;
    update_status(info, OSQP_UNSOLVED);
    info->rho_updates = 0;
    info->rho_estimate = settings->rho;

    workspace->data = data;
    workspace->linsys_solver = (LinSysSolver *)qdldl;
    workspace->rho_vec = (c_float *)buffers->work_rho_vec.ptr;
    workspace->rho_inv_vec = (c_float *)buffers->work_rho_inv_vec.ptr;
    workspace->constr_type = (c_int *)buffers->work_constr_type.ptr;
    workspace->x = (c_float *)buffers->work_x.ptr;
    workspace->y = (c_float *)buffers->work_y.ptr;
    workspace->z = (c_float *)buffers->work_z.ptr;
    workspace->xz_tilde = (c_float *)buffers->work_xz_tilde.ptr;
    workspace->x_prev = (c_float *)buffers->work_x_prev.ptr;
    workspace->z_prev = (c_float *)buffers->work_z_prev.ptr;
    workspace->Ax = (c_float *)buffers->work_Ax.ptr;
    workspace->Px = (c_float *)buffers->work_Px.ptr;
    workspace->Aty = (c_float *)buffers->work_Aty.ptr;
    workspace->delta_y = (c_float *)buffers->work_delta_y.ptr;
    workspace->Atdelta_y = (c_float *)buffers->work_Atdelta_y.ptr;
    workspace->delta_x = (c_float *)buffers->work_delta_x.ptr;
    workspace->Pdelta_x = (c_float *)buffers->work_Pdelta_x.ptr;
    workspace->Adelta_x = (c_float *)buffers->work_Adelta_x.ptr;
    workspace->D_temp = NULL;
    workspace->D_temp_A = NULL;
    workspace->E_temp = NULL;
    workspace->settings = settings;
    workspace->scaling = NULL;
    workspace->solution = solution;
    workspace->info = info;

    set_rho_vec(workspace);
    if (shape->m != 0) {
        memmove(buffers->qdldl_rho_inv_vec.ptr,
                workspace->rho_inv_vec,
                (size_t)layout->qdldl_rho_inv_vec.bytes);
    }
    native_status = form_KKT_into(
        pdata, adata, settings->sigma,
        (c_float *)buffers->qdldl_rho_inv_vec.ptr,
        kkt,
        (c_int *)buffers->qdldl_KKT_p.ptr,
        (c_int)(layout->qdldl_KKT_p.bytes / sizeof(c_int)),
        (c_int *)buffers->qdldl_KKT_i.ptr,
        (c_int)(layout->qdldl_KKT_i.bytes / sizeof(c_int)),
        (c_float *)buffers->qdldl_KKT_x.ptr,
        (c_int)(layout->qdldl_KKT_x.bytes / sizeof(c_float)),
        (c_int *)buffers->qdldl_PtoKKT.ptr,
        shape->p.nnz,
        (c_int *)buffers->qdldl_AtoKKT.ptr,
        shape->a.nnz,
        (c_int *)buffers->qdldl_Pdiag_idx.ptr,
        shape->n,
        &pdiag_n,
        (c_int *)buffers->qdldl_rhotoKKT.ptr,
        shape->m);
    if (native_status != 0) {
        return COKER_OSQP_INVALID_SHAPE;
    }

    native_status = init_linsys_solver_qdldl_into(
        qdldl, kkt, shape->n, shape->m, settings->sigma,
        linsys_l,
        (c_int *)buffers->qdldl_L_p.ptr,
        (c_int)(layout->qdldl_L_p.bytes / sizeof(c_int)),
        (c_int *)buffers->qdldl_L_i.ptr,
        (c_int)(layout->qdldl_L_i.bytes / sizeof(c_int)),
        (c_float *)buffers->qdldl_L_x.ptr,
        (c_int)(layout->qdldl_L_x.bytes / sizeof(c_float)),
        (QDLDL_float *)buffers->qdldl_D.ptr,
        (c_float *)buffers->qdldl_Dinv.ptr,
        (c_int *)buffers->qdldl_P.ptr,
        (c_float *)buffers->qdldl_bp.ptr,
        (c_float *)buffers->qdldl_sol.ptr,
        (c_float *)buffers->qdldl_rho_inv_vec.ptr,
        (c_int *)buffers->qdldl_Pdiag_idx.ptr,
        pdiag_n,
        (c_int *)buffers->qdldl_PtoKKT.ptr,
        (c_int *)buffers->qdldl_AtoKKT.ptr,
        (c_int *)buffers->qdldl_rhotoKKT.ptr,
        (QDLDL_int *)buffers->qdldl_etree.ptr,
        (QDLDL_int *)buffers->qdldl_Lnz.ptr,
        (QDLDL_int *)buffers->qdldl_iwork.ptr,
        (QDLDL_bool *)buffers->qdldl_bwork.ptr,
        (QDLDL_float *)buffers->qdldl_fwork.ptr);
    if (native_status != 0) {
        return COKER_OSQP_INVALID_SHAPE;
    }

    instance->pdata = pdata;
    instance->adata = adata;
    instance->data = data;
    instance->settings = settings;
    instance->scaling = NULL;
    instance->solution = solution;
    instance->info = info;
    instance->linsys_solver = qdldl;
    instance->qdldl = qdldl;
    instance->workspace = workspace;
    return COKER_OSQP_OK;
}

CokerOsqpStatus coker_osqp_update(CokerOsqpInstance *instance,
                                   const CokerOsqpNumericUpdate *update) {
    OSQPData *data;
    OSQPWorkspace *workspace;
    qdldl_solver *qdldl;
    c_int index;
    c_int native_status;

    if (!instance || !update) {
        return COKER_OSQP_INVALID_ARGUMENT;
    }
    if (!coker_osqp_numeric_update_is_locally_valid(update)) {
        return COKER_OSQP_INVALID_NUMERIC_UPDATE;
    }
    if (!coker_osqp_instance_is_bound(instance)) {
        return COKER_OSQP_NOT_BOUND;
    }

    data = (OSQPData *)instance->data;
    workspace = (OSQPWorkspace *)instance->workspace;
    qdldl = (qdldl_solver *)instance->qdldl;
    if (update->p_nnz != data->P->nzmax ||
        update->a_nnz != data->A->nzmax || update->q_len != data->n ||
        update->l_len != data->m || update->u_len != data->m ||
        !coker_osqp_values_are_finite(update->p_x, update->p_nnz) ||
        !coker_osqp_values_are_finite(update->a_x, update->a_nnz) ||
        !coker_osqp_values_are_finite(update->q, update->q_len) ||
        !coker_osqp_values_are_finite(update->l, update->l_len) ||
        !coker_osqp_values_are_finite(update->u, update->u_len)) {
        return COKER_OSQP_INVALID_NUMERIC_UPDATE;
    }
    for (index = 0; index < data->m; ++index) {
        if (update->l[index] > update->u[index]) {
            return COKER_OSQP_INVALID_NUMERIC_UPDATE;
        }
    }

    if (data->P->nzmax != 0) {
        memmove(data->P->x, update->p_x,
                (size_t)data->P->nzmax * sizeof(*data->P->x));
    }
    if (data->A->nzmax != 0) {
        memmove(data->A->x, update->a_x,
                (size_t)data->A->nzmax * sizeof(*data->A->x));
    }
    memmove(data->q, update->q, (size_t)data->n * sizeof(*data->q));
    if (data->m != 0) {
        memmove(data->l, update->l, (size_t)data->m * sizeof(*data->l));
        memmove(data->u, update->u, (size_t)data->m * sizeof(*data->u));
    }

    native_status = qdldl->update_matrices(qdldl, data->P, data->A);
    if (native_status == 0) {
        native_status = update_rho_vec(workspace);
    }
    reset_info(workspace->info);
    if (native_status != 0) {
        memset(instance, 0, sizeof(*instance));
        return COKER_OSQP_INVALID_NUMERIC_UPDATE;
    }
    return COKER_OSQP_OK;
}

CokerOsqpStatus coker_osqp_solve(CokerOsqpInstance *instance,
                                  CokerOsqpSolveStatus *solve_status) {
    OSQPWorkspace *workspace;
    c_int native_status;

    if (!instance || !solve_status) {
        return COKER_OSQP_INVALID_ARGUMENT;
    }
    *solve_status = COKER_OSQP_SOLVE_UNSOLVED;
    if (!coker_osqp_instance_is_bound(instance)) {
        return COKER_OSQP_NOT_BOUND;
    }

    workspace = (OSQPWorkspace *)instance->workspace;
    native_status = osqp_solve(workspace);
    *solve_status =
        coker_osqp_solve_status_from_native(workspace->info->status_val);
    return native_status == 0 ? COKER_OSQP_OK : COKER_OSQP_INVALID_NUMERIC_UPDATE;
}

CokerOsqpStatus coker_osqp_solution(const CokerOsqpInstance *instance,
                                     CokerOsqpSolution *solution) {
    const OSQPData *data;
    const OSQPSolution *native_solution;
    const OSQPInfo *info;

    if (!instance || !solution) {
        return COKER_OSQP_INVALID_ARGUMENT;
    }
    solution->primal = NULL;
    solution->primal_len = 0;
    solution->dual = NULL;
    solution->dual_len = 0;
    solution->status = COKER_OSQP_SOLVE_UNSOLVED;
    solution->iterations = 0;
    solution->primal_residual = 0.0f;
    solution->dual_residual = 0.0f;
    if (!coker_osqp_instance_is_bound(instance)) {
        return COKER_OSQP_NOT_BOUND;
    }

    data = (const OSQPData *)instance->data;
    native_solution = (const OSQPSolution *)instance->solution;
    info = (const OSQPInfo *)instance->info;
    solution->primal = native_solution->x;
    solution->primal_len = data->n;
    solution->dual = native_solution->y;
    solution->dual_len = data->m;
    solution->status = coker_osqp_solve_status_from_native(info->status_val);
    solution->iterations = info->iter;
    solution->primal_residual = info->pri_res;
    solution->dual_residual = info->dua_res;
    return COKER_OSQP_OK;
}
