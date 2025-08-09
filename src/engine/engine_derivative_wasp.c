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
//     DyDq: (nx x nv)
//     DyDv: (nv x nv)
//     DyDa: (nv x na)
//     DyDu: (nv x nu)
//     DsDq: (ns x nv)
//     DsDv: (ns x nv)
//     DsDa: (ns x na)
//     DsDu: (ns x nu)
//   single-letter shortcuts:
//     inputs: q=qpos, v=qvel, a=activatoin, u=ctrl
//     outputs: y=next_state (concatenated next qpos, qvel, activation), s=sensordata
void mjd_stepWASP(const mjModel* m, mjData* d,
                mjtNum eps, mjtByte flg_centered,
                mjWASPCache* DyDq, mjWASPCache* DyDv, mjWASPCache* DyDa,
                mjWASPCache* DyDu,
                mjWASPCache* DsDq, mjWASPCache* DsDv, mjWASPCache* DsDa,
                mjWASPCache* DsDu) {

    int nq = m->nq, nv = m->nv, na = m->na, nu = m->nu, ns = m->nsensordata;
    int ndx = 2*nv+na;  // col length of Dy Jacobians
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
    if (DyDu || DsDu) {
        size_t iu = DyDu->i;
        int limited = m->actuator_ctrllimited[iu];
        // nudge forward, if possible given ctrlrange
        int nudge_fwd = !limited || inRange(ctrl[iu], ctrl[iu]+eps, m->actuator_ctrlrange+2*iu);
        if (nudge_fwd) {
            // nudge forward
            d->ctrl[iu] += eps;

            // step, get nudged output
            mj_stepSkip(m, d, mjSTAGE_VEL, skipsensor);
            getState(m, d, next_plus, sensor_plus);

            // reset
            mj_setState(m, d, fullstate, restore_spec);
        }

        // nudge backward, if possible given ctrlrange
        int nudge_back = (flg_centered || !nudge_fwd) &&
                         (!limited || inRange(ctrl[iu]-eps, ctrl[iu], m->actuator_ctrlrange+2*iu));
        if (nudge_back) {
            // nudge backward
            d->ctrl[iu] -= eps;

            // step, get nudged output
            mj_stepSkip(m, d, mjSTAGE_VEL, skipsensor);
            getState(m, d, next_minus, sensor_minus);

            // reset
            mj_setState(m, d, fullstate, restore_spec);
        }

        // difference states
        if (DyDu) {
            clampedStateDiff(m, DyDu->fi, next, nudge_fwd ? next_plus : NULL,
                             nudge_back ? next_minus : NULL, eps);
        }

        // difference sensors
        if (DsDu) {
            clampedDiff(DsDu->fi, sensor, nudge_fwd ? sensor_plus : NULL,
                        nudge_back ? sensor_minus : NULL, eps, ns);
        }
    }

    // wasp-difference activations: skip=mjSTAGE_VEL
    if (DyDa || DsDa) {
        size_t ia = DyDa->i;
        // nudge forward
        d->act[ia] += eps;

        // step, get nudged output
        mj_stepSkip(m, d, mjSTAGE_VEL, skipsensor);
        getState(m, d, next_plus, sensor_plus);

        // reset
        mj_setState(m, d, fullstate, restore_spec);

        // nudge backward
        if (flg_centered) {
            // nudge backward
            d->act[ia] -= eps;

            // step, get nudged output
            mj_stepSkip(m, d, mjSTAGE_VEL, skipsensor);
            getState(m, d, next_minus, sensor_minus);

            // reset
            mj_setState(m, d, fullstate, restore_spec);
        }
        // difference states
        if (DyDa) {
            if (!flg_centered) {
                stateDiff(m, DyDa->fi, next, next_plus, eps);
            } else {
                stateDiff(m, DyDa->fi, next_minus, next_plus, 2*eps);
            }
        }

        // difference sensors
        if (DsDa) {
            if (!flg_centered) {
                diff(DsDa->fi, sensor, sensor_plus, eps, ns);
            } else {
                diff(DsDa->fi, sensor_minus, sensor_plus, 2*eps, ns);
            }
        }
    }

    // wasp-difference velocities: skip=mjSTAGE_POS
    if (DyDv || DsDv) {
        size_t iv = DyDv->i;
        // nudge forward
        d->qvel[iv] += eps;

        // step, get nudged output
        mj_stepSkip(m, d, mjSTAGE_POS, skipsensor);
        getState(m, d, next_plus, sensor_plus);

        // reset
        mj_setState(m, d, fullstate, restore_spec);

        // nudge backward
        if (flg_centered) {
            // nudge
            d->qvel[iv] -= eps;

            // step, get nudged output
            mj_stepSkip(m, d, mjSTAGE_POS, skipsensor);
            getState(m, d, next_minus, sensor_minus);

            // reset
            mj_setState(m, d, fullstate, restore_spec);
        }
        // difference states
        if (DyDv) {
            if (!flg_centered) {
                stateDiff(m, DyDv->fi, next, next_plus, eps);
            } else {
                stateDiff(m, DyDv->fi, next_minus, next_plus, 2*eps);
            }
        }

        // difference sensors
        if (DsDv) {
            if (!flg_centered) {
                diff(DsDv->fi, sensor, sensor_plus, eps, ns);
            } else {
                diff(DsDv->fi, sensor_minus, sensor_plus, 2*eps, ns);
            }
        }
    }

    // wasp-difference positions: skip=mjSTAGE_NONE
    if (DyDq || DsDq) {
        size_t iq = DyDq->i;
        mjtNum *dpos  = mj_stackAllocNum(d, nv);  // allocate position perturbation
        // nudge forward
        mju_zero(dpos, nv);
        dpos[iq] = 1;
        mj_integratePos(m, d->qpos, dpos, eps);

        // step, get nudged output
        mj_stepSkip(m, d, mjSTAGE_NONE, skipsensor);
        getState(m, d, next_plus, sensor_plus);

        // reset
        mj_setState(m, d, fullstate, restore_spec);

        // nudge backward
        if (flg_centered) {
            // nudge backward
            mju_zero(dpos, nv);
            dpos[iq] = 1;
            mj_integratePos(m, d->qpos, dpos, -eps);

            // step, get nudged output
            mj_stepSkip(m, d, mjSTAGE_NONE, skipsensor);
            getState(m, d, next_minus, sensor_minus);

            // reset
            mj_setState(m, d, fullstate, restore_spec);
        }

        // difference states
        if (DyDq) {
            if (!flg_centered) {
                stateDiff(m, DyDq->fi, next, next_plus, eps);
            } else {
                stateDiff(m, DyDq->fi, next_minus, next_plus, 2*eps);
            }
        }

        // difference sensors
        if (DsDq) {
            if (!flg_centered) {
                diff(DsDq->fi, sensor, sensor_plus, eps, ns);
            } else {
                diff(DsDq->fi, sensor_minus, sensor_plus, 2*eps, ns);
            }
        }
    }
    mj_freeStack(d);
}

// dim(res) = m x n
// D = F_hat * C1^T + fi * C2^T
// F_hat = D * Delta_X
static void waspUpdate(mjData* d, mjtNum* res, mjWASPCache* cache, int m, int n){
    size_t i = cache->i;
    mjtNum* tmp =  mj_stackAllocNum(d, m*n);
    mju_mulMatMatT(res, cache->F_hat, (cache->C1)+i*n*n, m, n, n);
    mju_mulMatMatT(tmp, cache->fi, (cache->C2)+i, m, 1, n);
    mju_addToMat(res, tmp, m, m);
    mju_mulMatMat(cache->F_hat, res, cache->Delta_X, n, n, n);
    cache->i = (i+1)%n;
}



// wasp differenced transition matrices (control theory notation)
//   d(x_next) = A*Dx + B*Du
//   d(sensor) = C*Dx + D*Du
//   required output matrix dimensions:
//      nx = nq + nv + na (position + velocity + activation)
//      A: model Jacobian wrt state, (nx * nx)
//      B: model Jacobian wrt control, (nx * nu)
//      C: sensor Jacobian wrt state, (ns * nx)
//      D: sensor Jacobian wrt control, (nx * nu)
void mjd_transitionWASP(const mjModel* m, mjData* d, mjtNum eps, mjtByte flg_centered,
                        mjtNum* A, mjtNum* B, mjtNum* C, mjtNum* D,
                        mjWASPCache* DyDq_cache, mjWASPCache* DyDv_cache, mjWASPCache* DyDa_cache,
                        mjWASPCache* DyDu_cache,
                        mjWASPCache* DsDq_cache, mjWASPCache* DsDv_cache, mjWASPCache* DsDa_cache,
                        mjWASPCache* DsDu_cache){

    if (m->opt.integrator == mjINT_RK4) {
        mjERROR("RK4 integrator is not supported");
    }

    int nv = m->nv, na = m->na, nu = m->nu, ns = m->nsensordata;
    int ndx = 2*nv+na;  // row length of state Jacobians

    // finite difference on the specific dimensions
    mjd_stepWASP(m, d, eps, flg_centered, DyDq_cache, DyDv_cache, DyDa_cache,
                 DyDu_cache,
                 DsDq_cache, DsDv_cache, DsDa_cache,
                 DsDu_cache);

    if (A) {
        if (DyDq_cache){
            waspUpdate(d, A, DyDq_cache, ndx, nv);
        }
        if (DyDv_cache){
            waspUpdate(d, A+nv, DyDv_cache, ndx, nv);
        }
        if (DyDa_cache){
            waspUpdate(d, A+2*nv, DyDa_cache, ndx, na);
        }

    }
    if (B&&DyDu_cache) {
        waspUpdate(d, B, DyDu_cache, ndx, nu);
    }
    if (C) {
        if (DsDq_cache){
            waspUpdate(d, C, DsDq_cache, ns, nv);
        }
        if (DsDv_cache){
            waspUpdate(d, C+nv, DsDv_cache, ns, nv);
        }
        if (DsDa_cache){
            waspUpdate(d, C+2*nv, DsDa_cache, ns, na);
        }
    }
    if (D&&DsDu_cache) {
        waspUpdate(d, D, DsDu_cache, ns, nu);
    }

    mj_freeStack(d);
}
