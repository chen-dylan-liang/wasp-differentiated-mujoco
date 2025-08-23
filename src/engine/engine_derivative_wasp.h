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
    // reset wasp cache
    MJAPI void mj_resetWASPCache(mjWASPCache* cache, int n, int m, mjtByte identity_basis);

    // delete wasp cache
    MJAPI void mj_deleteWASPCache(mjWASPCache* cache);

    // allocate wasp cache
    MJAPI mjWASPCache* mj_newWASPCache(int n, int m, mjtByte identity_basis);

    // copy wasp cache
    MJAPI void mj_copyWASPCache(int n, int m, mjWASPCache* dest, const mjWASPCache* src);


    // wasp differenced transition matrices (control theory notation)
    MJAPI void mjd_transitionWASP(const mjModel* m, mjData* d, mjtNum eps, mjtByte flg_centered,
                                  mjtNum q_dtheta, mjtNum q_dell, int q_max_n,
                                  mjtNum v_dtheta, mjtNum v_dell, int v_max_n,
                                  mjtNum a_dtheta, mjtNum a_dell, int a_max_n,
                                  mjtNum u_dtheta, mjtNum u_dell, int u_max_n,
                                  mjtNum* A, mjtNum* B, mjtNum* C, mjtNum* D,
                                  mjWASPCache* DyDq_cache, mjWASPCache* DyDv_cache, mjWASPCache* DyDa_cache,
                                  mjWASPCache* DyDu_cache,
                                  mjWASPCache* DsDq_cache, mjWASPCache* DsDv_cache, mjWASPCache* DsDa_cache,
                                  mjWASPCache* DsDu_cache);

#ifdef __cplusplus
    }
#endif

#endif // MUJOCO_SRC_ENGINE_ENGINE_DERIVATIVE_WASP_H_
