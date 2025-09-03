//
// Created by dylan on 7/26/25.
//

#ifndef MUJOCO_SRC_ENGINE_ENGINE_DERIVATIVE_WASP_H_
#define MUJOCO_SRC_ENGINE_ENGINE_DERIVATIVE_WASP_H_

#include <mujoco/mjdata.h>
#include <mujoco/mjexport.h>
#include <mujoco/mjmodel.h>
#include "engine_derivative_fd.h"
#include "engine/engine_util_solve.h"
#include <stdlib.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif
    // allocate wasp cache basis
    MJAPI mjWASPBasis* mj_newWASPBasis(int n, mjtByte identity_basis);

    // delete wasp cache
    MJAPI void mj_deleteWASPCache(mjWASPCache* cache);

    // delete wasp basis
    MJAPI void mj_deleteWASPBasis(mjWASPBasis* basis);

    // allocate wasp cache
    MJAPI mjWASPCache *mj_newWASPCache(int n, int m);

    // zero wasp cache (zero Fhat, fi, and i)
    MJAPI void mj_zeroWASPCache(mjWASPCache* cache, int n, int m);


    // wasp differenced transition matrices (control theory notation)
    MJAPI int mjd_transitionWASP(const mjModel* m,
                        const mjWASPBasis* q_basis, const mjWASPBasis* v_basis,
                        const mjWASPBasis* a_basis, const mjWASPBasis* u_basis,
                        mjData* d, mjtNum eps, mjtByte flg_centered,
                        mjtNum q_dtheta, mjtNum q_dell, int q_min_n,
                        mjtNum v_dtheta, mjtNum v_dell, int v_min_n,
                        mjtNum a_dtheta, mjtNum a_dell, int a_min_n,
                        mjtNum u_dtheta, mjtNum u_dell, int u_min_n,
                        mjtNum* A, mjtNum* B, mjtNum* C, mjtNum* D,
                        mjWASPCache* DyDq, mjWASPCache* DyDv, mjWASPCache* DyDa,
                        mjWASPCache* DyDu,
                        mjWASPCache* DsDq, mjWASPCache* DsDv, mjWASPCache* DsDa,
                        mjWASPCache* DsDu);

    // per thread wasp differenced transition matrices (control theory notation)
    MJAPI int mjd_transitionWASPOneThread(const mjModel *m, const mjWASPBasis* basis, mjData *d, mjtNum eps, mjtByte flg_centered,
                                  mjtNum dtheta, mjtNum dell, int min_n,
                                  mjtNum *derivT,
                                  mjWASPCache *cache, mjPartialDerivativeType type);
    // copy wasp cache
    MJAPI void mj_copyWASPCache(mjWASPCache *dest, const mjWASPCache* src, int n, int m);

#ifdef __cplusplus
    }
#endif

#endif // MUJOCO_SRC_ENGINE_ENGINE_DERIVATIVE_WASP_H_
