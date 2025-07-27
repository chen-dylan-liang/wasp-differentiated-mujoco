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
    // wasp differenced transition matrices (control theory notation)
    MJAPI void mjd_transitionWASP(const mjModel* m, mjData* d,
                                mjtNum eps, mjtByte centered, int32_t* wasp_idx,
                                mjtNum* A, mjtNum* B, mjtNum* C, mjtNum* D);


#ifdef __cplusplus
    }
#endif

#endif // MUJOCO_SRC_ENGINE_ENGINE_DERIVATIVE_WASP_H_
