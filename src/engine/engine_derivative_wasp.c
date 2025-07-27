//
// Created by dylan on 7/26/25.
//
#include "engine/engine_derivative_wasp.h"
#include "engine/engine_derivative_fd.c"

#include <stddef.h>
#include <string.h>

#include <mujoco/mjdata.h>
#include <mujoco/mjmacro.h>
#include <mujoco/mjmodel.h>
#include "engine/engine_forward.h"
#include "engine/engine_io.h"
#include "engine/engine_inverse.h"
#include "engine/engine_macro.h"
#include "engine/engine_support.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_errmem.h"
#include "engine/engine_util_misc.h"



//   wasp differentiate Jacobian of  (next_state, sensors) = mj_step(state, control)
//   all outputs are optional
//   output dimensions (transposed w.r.t Control Theory convention):
//     DyDq: (nv x dim_state)
//     DyDv: (nv x dim_state)
//     DyDa: (na x dim_state)
//     DyDu: (nu x dim_state)
//     DsDq: (nv x dim_sensor)
//     DsDv: (nv x dim_sensor)
//     DsDa: (na x dim_sensor)
//     DsDu: (nu x dim_sensor)
//   single-letter shortcuts:
//     inputs: q=qpos, v=qvel, a=activatoin, u=ctrl
//     outputs: y=next_state (concatenated next qpos, qvel, act), s=sensordata
void mjd_stepWASP(const mjModel* m, mjData* d,
                mjtNum eps, mjtByte flg_centered, int32_t* wasp_idx,
                mjtNum* DyDq, mjtNum* DyDv, mjtNum* DyDa, mjtNum* DyDu,
                mjtNum* DsDq, mjtNum* DsDv, mjtNum* DsDa, mjtNum* DsDu) {

    int nq = m->nq, nv = m->nv, na = m->na, nu = m->nu, ns = m->nsensordata;
    int ndx = 2*nv+na;  // row length of Dy Jacobians
    mj_markStack(d);

    // state to restore after finite differencing
    unsigned int restore_spec = mjSTATE_FULLPHYSICS | mjSTATE_CTRL;
    restore_spec |= mjDISABLED(mjDSBL_WARMSTART) ? 0 : mjSTATE_WARMSTART;

    mjtNum *fullstate  = mj_stackAllocNum(d, mj_stateSize(m, restore_spec));
    mjtNum *state      = mj_stackAllocNum(d, nq+nv+na);  // current state
    mjtNum *next       = mj_stackAllocNum(d, nq+nv+na);  // next state
    mjtNum *next_plus  = mj_stackAllocNum(d, nq+nv+na);  // forward-nudged next state
    mjtNum *next_minus = mj_stackAllocNum(d, nq+nv+na);  // backward-nudged next state

    // sensors
    int skipsensor = !DsDq && !DsDv && !DsDa && !DsDu;
    mjtNum *sensor       = skipsensor ? NULL : mj_stackAllocNum(d, ns);  // sensor values
    mjtNum *sensor_plus  = skipsensor ? NULL : mj_stackAllocNum(d, ns);  // forward-nudged sensors
    mjtNum *sensor_minus = skipsensor ? NULL : mj_stackAllocNum(d, ns);  // backward-nudged sensors

    // controls
    mjtNum *ctrl = mj_stackAllocNum(d, nu);

    // save current inputs
    mj_getState(m, d, fullstate, restore_spec);
    mju_copy(ctrl, d->ctrl, nu);
    getState(m, d, state, NULL);

    // step input
    mj_stepSkip(m, d, mjSTAGE_NONE, skipsensor);

    // save output
    getState(m, d, next, sensor);

    // restore input
    mj_setState(m, d, fullstate, restore_spec);

    // wasp-difference controls: skip=mjSTAGE_VEL, handle ctrl at range limits
    int32_t i = *wasp_idx;
    if (DyDu || DsDu) {
        // difference states
        if (DyDu) {
            clampedStateDiff(m, DyDu+i*ndx, next, nudge_fwd ? next_plus : NULL,
                             nudge_back ? next_minus : NULL, eps);
        }

        // difference sensors
        if (DsDu) {
            clampedDiff(DsDu+i*ns, sensor, nudge_fwd ? sensor_plus : NULL,
                        nudge_back ? sensor_minus : NULL, eps, ns);
        }
    }
    // wasp-difference activations: skip=mjSTAGE_VEL
    if (DyDa || DsDa) {
        // difference states
        if (DyDa) {
            if (!flg_centered) {
                stateDiff(m, DyDa+i*ndx, next, next_plus, eps);
            } else {
                stateDiff(m, DyDa+i*ndx, next_minus, next_plus, 2*eps);
            }
        }

        // difference sensors
        if (DsDa) {
            if (!flg_centered) {
                diff(DsDa+i*ns, sensor, sensor_plus, eps, ns);
            } else {
                diff(DsDa+i*ns, sensor_minus, sensor_plus, 2*eps, ns);
            }
        }
    }
    // wasp-difference velocities: skip=mjSTAGE_POS
    if (DyDv || DsDv) {
        // difference states
        if (DyDv) {
            if (!flg_centered) {
                stateDiff(m, DyDv+i*ndx, next, next_plus, eps);
            } else {
                stateDiff(m, DyDv+i*ndx, next_minus, next_plus, 2*eps);
            }
        }

        // difference sensors
        if (DsDv) {
            if (!flg_centered) {
                diff(DsDv+i*ns, sensor, sensor_plus, eps, ns);
            } else {
                diff(DsDv+i*ns, sensor_minus, sensor_plus, 2*eps, ns);
            }
        }
    }
    // wasp-difference positions: skip=mjSTAGE_NONE
    if (DyDq || DsDq) {
        // difference states
        if (DyDq) {
            if (!flg_centered) {
                stateDiff(m, DyDq+i*ndx, next, next_plus, eps);
            } else {
                stateDiff(m, DyDq+i*ndx, next_minus, next_plus, 2*eps);
            }
        }

        // difference sensors
        if (DsDq) {
            if (!flg_centered) {
                diff(DsDq+i*ns, sensor, sensor_plus, eps, ns);
            } else {
                diff(DsDq+i*ns, sensor_minus, sensor_plus, 2*eps, ns);
            }
        }
    }
    mj_freeStack(d);
}






// wasp differenced transition matrices (control theory notation)
//   d(x_next) = A*dx + B*du
//   d(sensor) = C*dx + D*du
//   required output matrix dimensions:
//      A: model Jacobian wrt state, (dim_state* dim_state)
//      B: model Jacobian wrt action, (dim_state * dim_action)
//      C: sensor Jacobian wrt state, (dim_sensor * dim_state)
//      D: sensor Jacobian wrt action, (dim_sensor * dim_action)
void mjd_transitionWASP(const mjModel* m, mjData* d, mjtNum eps, mjtByte flg_centered, int32_t* wasp_idx,
                      mjtNum* A, mjtNum* B, mjtNum* C, mjtNum* D){

    if (m->opt.integrator == mjINT_RK4) {
        mjERROR("RK4 integrator is not supported");
    }

    int nv = m->nv, na = m->na, nu = m->nu, ns = m->nsensordata;
    int ndx = 2*nv+na;  // row length of state Jacobians

    // stepFD() offset pointers, initialised to NULL
    mjtNum *DyDq, *DyDv, *DyDa, *DsDq, *DsDv, *DsDa;
    DyDq = DyDv = DyDa = DsDq = DsDv = DsDa = NULL;

    mj_markStack(d);

    // allocate transposed matrices
    mjtNum *AT = A ? mj_stackAllocNum(d, ndx*ndx) : NULL;  // state-transition matrix   (transposed)
    mjtNum *BT = B ? mj_stackAllocNum(d, nu*ndx) : NULL;   // control-transition matrix (transposed)
    mjtNum *CT = C ? mj_stackAllocNum(d, ndx*ns) : NULL;   // state-observation matrix   (transposed)
    mjtNum *DT = D ? mj_stackAllocNum(d, nu*ns) : NULL;    // control-observation matrix (transposed)

    // set offset pointers
    if (A) {
        DyDq = AT;
        DyDv = AT+ndx*nv;
        DyDa = AT+ndx*2*nv;
    }

    if (C) {
        DsDq = CT;
        DsDv = CT + ns*nv;
        DsDa = CT + ns*2*nv;
    }

    // get Jacobians
    mjd_stepWASP(m, d, eps, flg_centered, wasp_idx,DyDq, DyDv, DyDa, BT, DsDq, DsDv, DsDa, DT);


    // transpose
    if (A) mju_transpose(A, AT, ndx, ndx);
    if (B) mju_transpose(B, BT, nu, ndx);
    if (C) mju_transpose(C, CT, ndx, ns);
    if (D) mju_transpose(D, DT, nu, ns);

    mj_freeStack(d);
}