//
// Created by dylan on 7/26/25.
//

#ifndef MUJOCO_SRC_ENGINE_ENGINE_DERIVATIVE_WASP_H_
#define MUJOCO_SRC_ENGINE_ENGINE_DERIVATIVE_WASP_H_

#include <mujoco/mjdata.h>
#include <mujoco/mjexport.h>
#include <mujoco/mjmodel.h>

#ifdef __cplusplus
extern "C" {
#endif
    struct mjWASPCache_{
        mjtNum* Delta_X;
        mjtNum* C1;
        mjtNum* C2;
        mjtNum* F_hat;
        mjtNum* fi;
        size_t i;
    };
    typedef struct mjWASPCache_ mjWASPCache;

    // wasp differenced transition matrices (control theory notation)
    MJAPI void mjd_transitionWASP(const mjModel* m, mjData* d, mjtNum eps, mjtByte flg_centered,
                                  mjtNum* A, mjtNum* B, mjtNum* C, mjtNum* D,
                                  mjWASPCache* DyDq_cache, mjWASPCache* DyDv_cache, mjWASPCache* DyDa_cache,
                                  mjWASPCache* DyDu_cache,
                                  mjWASPCache* DsDq_cache, mjWASPCache* DsDv_cache, mjWASPCache* DsDa_cache,
                                  mjWASPCache* DsDu_cache);

#ifdef __cplusplus
    }
#endif

#endif // MUJOCO_SRC_ENGINE_ENGINE_DERIVATIVE_WASP_H_
