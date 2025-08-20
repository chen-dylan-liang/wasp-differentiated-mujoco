//
// Created by dylan on 7/26/25.
//
#include "engine/engine_derivative_wasp.h"


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


//--------------------------- static utility functions copied directly from engine_derivative_fd.c--------------------------------

// get state=[qpos; qvel; act] and optionally sensordata
static void getState(const mjModel* m, const mjData* d, mjtNum* state, mjtNum* sensordata) {
    mj_getState(m, d, state, mjSTATE_PHYSICS);
    if (sensordata) {
        mju_copy(sensordata, d->sensordata, m->nsensordata);
    }
}



// dx = (x2 - x1) / h
static void diff(mjtNum* restrict dx, const mjtNum* x1, const mjtNum* x2, mjtNum h, int n) {
    mjtNum inv_h = 1/h;
    for (int i=0; i < n; i++) {
        dx[i] = inv_h * (x2[i] - x1[i]);
    }
}



// finite-difference two state vectors ds = (s2 - s1) / h
static void stateDiff(const mjModel* m, mjtNum* ds, const mjtNum* s1, const mjtNum* s2, mjtNum h) {
    int nq = m->nq, nv = m->nv, na = m->na;

    if (nq == nv) {
        diff(ds, s1, s2, h, nq+nv+na);
    } else {
        mj_differentiatePos(m, ds, h, s1, s2);
        diff(ds+nv, s1+nq, s2+nq, h, nv+na);
    }
}



// finite-difference two vectors, forward, backward or centered
static void clampedDiff(mjtNum* dx, const mjtNum* x, const mjtNum* x_plus, const mjtNum* x_minus,
                        mjtNum h, int nx) {
    if (x_plus && !x_minus) {
        // forward differencing
        diff(dx, x, x_plus, h, nx);
    } else if (!x_plus && x_minus) {
        // backward differencing
        diff(dx, x_minus, x, h, nx);
    } else if (x_plus && x_minus) {
        // centered differencing
        diff(dx, x_plus, x_minus, 2*h, nx);
    } else {
        // differencing failed, write zeros
        mju_zero(dx, nx);
    }
}

static mjtByte closeEnough(const mjtNum* a, const mjtNum* b, int n, mjtNum dell, mjtNum dtheta){
    mjtNum dot = mju_dot(a, b, n);
    mjtNum a_norm = mju_norm(a, n);
    mjtNum b_norm = mju_norm(b, n);
    if (mju_abs(dot/a_norm/b_norm-1)>dtheta)
        return 0;
    if (mju_min(mju_abs(a_norm/b_norm-1), mju_abs(b_norm/a_norm-1))>dell)
        return 0;
    return 1;
}

// finite-difference two state vectors, forward, backward or centered
static void clampedStateDiff(const mjModel* m, mjtNum* ds, const mjtNum* s, const mjtNum* s_plus,
                             const mjtNum* s_minus, mjtNum h) {
    if (s_plus && !s_minus) {
        // forward differencing
        stateDiff(m, ds, s, s_plus, h);
    } else if (!s_plus && s_minus) {
        // backward differencing
        stateDiff(m, ds, s_minus, s, h);
    } else if (s_plus && s_minus) {
        // centered differencing
        stateDiff(m, ds, s_minus, s_plus, 2*h);
    } else {
        // differencing failed, write zeros
        mju_zero(ds, 2*m->nv + m->na);
    }
}

// check if two numbers are inside a given range
static int inRange(const mjtNum x1, const mjtNum x2, const mjtNum* range) {
    return x1 >= range[0] && x1 <= range[1] &&
           x2 >= range[0] && x2 <= range[1];
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

static void mjd_stepWASPDu(const mjModel* m,
                           const mjtNum* fullstate,
                           const mjtNum* next,
                           const mjtNum* sensor,
                           const mjtNum* ctrl,
                           mjData* d,
                           mjtNum eps, mjtByte flg_centered,
                           int skipsensor, unsigned int restore_spec,
                           mjtNum dtheta, mjtNum dell, int max_n,
                           mjtNum* next_plus, mjtNum* next_minus,  mjtNum* sensor_plus, mjtNum* sensor_minus,
                           mjtNum* y_res,  mjtNum* s_res,
                           mjtNum* y_fi, mjtNum* s_fi,
                           mjWASPCache* y_cache, mjWASPCache* s_cache){
    for(int i=0; i<max_n ;i++){
        size_t iu = y_cache->i;
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
        mjtByte y_accurate=1, s_accurate=1;
        if (y_cache) {
            mju_copy(y_fi, y_cache->fi, 2*m->nv+m->na);
            clampedStateDiff(m, y_cache->fi, next, nudge_fwd ? next_plus : NULL,
                             nudge_back ? next_minus : NULL, eps);
            waspUpdate(d, y_res, y_cache, 2*m->nv+m->na, m->nu);
            y_accurate=closeEnough(y_fi, y_cache->fi, 2*m->nv+m->na, dell, dtheta);
        }

        // difference sensors
        if (s_cache) {
            mju_copy(s_fi, s_cache->fi, 2*m->nsensordata);
            clampedDiff(s_cache->fi, sensor, nudge_fwd ? sensor_plus : NULL,
                        nudge_back ? sensor_minus : NULL, eps,  m->nsensordata);
            waspUpdate(d, s_res, s_cache, m->nsensordata, m->nu);
            s_accurate=closeEnough(s_fi, s_cache->fi, m->nsensordata, dell, dtheta);
        }
        if(y_accurate&s_accurate) break;
    }
}

static void mjd_stepWASPDv(const mjModel* m,
                           const mjtNum* fullstate,
                           const mjtNum* next,
                           const mjtNum* sensor,
                           mjData* d,
                           mjtNum eps, mjtByte flg_centered,
                           int skipsensor, unsigned int restore_spec,
                           mjtNum dtheta, mjtNum dell, int max_n,
                           mjtNum* next_plus, mjtNum* next_minus,  mjtNum* sensor_plus, mjtNum* sensor_minus,
                           mjtNum* y_res,  mjtNum* s_res,
                           mjtNum* y_fi, mjtNum* s_fi,
                           mjWASPCache* y_cache, mjWASPCache* s_cache){
    for(int i=0; i<max_n ;i++){
    size_t iv = y_cache->i;
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
    mjtByte y_accurate=1, s_accurate=1;
    if (y_cache) {
        mju_copy(y_fi, y_cache->fi, 2*m->nv+m->na);
        if (!flg_centered) {
            stateDiff(m, y_cache->fi, next, next_plus, eps);
        } else {
            stateDiff(m, y_cache->fi, next_minus, next_plus, 2*eps);
        }
        waspUpdate(d, y_res+m->nv, y_cache, 2*m->nv+m->na, m->nv);
        y_accurate=closeEnough(y_fi, y_cache->fi, 2*m->nv+m->na,dell, dtheta);
    }

    // difference sensors
    if (s_cache) {
        mju_copy(s_fi, s_cache->fi, 2*m->nsensordata);
        if (!flg_centered) {
            diff(s_cache->fi, sensor, sensor_plus, eps, m->nsensordata);
        } else {
            diff(s_cache->fi, sensor_minus, sensor_plus, 2*eps, m->nsensordata);
        }
        waspUpdate(d, s_res+m->nv, s_cache, m->nsensordata,  m->nv);
        s_accurate=closeEnough(s_fi, s_cache->fi, m->nsensordata,dell, dtheta);
    }
        if(y_accurate&s_accurate) break;
    }

}

static void mjd_stepWASPDa(const mjModel* m,
                           const mjtNum* fullstate,
                           const mjtNum* next,
                           const mjtNum* sensor,
                           mjData* d,
                           mjtNum eps, mjtByte flg_centered,
                           int skipsensor, unsigned int restore_spec,
                           mjtNum dtheta, mjtNum dell, int max_n,
                           mjtNum* next_plus, mjtNum* next_minus,  mjtNum* sensor_plus, mjtNum* sensor_minus,
                           mjtNum* y_res,  mjtNum* s_res,
                           mjtNum* y_fi, mjtNum* s_fi,
                           mjWASPCache* y_cache, mjWASPCache* s_cache){
    for(int i=0; i<max_n ;i++){
    size_t ia = y_cache->i;
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
    mjtByte y_accurate=1, s_accurate=1;
    if (y_cache) {
        mju_copy(y_fi, y_cache->fi, 2*m->nv+m->na);
        if (!flg_centered) {
            stateDiff(m, y_cache->fi, next, next_plus, eps);
        } else {
            stateDiff(m, y_cache->fi, next_minus, next_plus, 2*eps);
        }
        waspUpdate(d, y_res+2*m->nv, y_cache, 2*m->nv+m->na, m->na);
        y_accurate=closeEnough(y_fi, y_cache->fi, 2*m->nv+m->na,dell, dtheta);
    }

    // difference sensors
    if (s_cache) {
        mju_copy(s_fi, s_cache->fi, 2*m->nsensordata);
        if (!flg_centered) {
            diff(s_cache->fi, sensor, sensor_plus, eps, m->nsensordata);
        } else {
            diff(s_cache->fi, sensor_minus, sensor_plus, 2*eps, m->nsensordata);
        }
        waspUpdate(d, s_res+2*m->nv, s_cache, m->nsensordata, m->na);
        s_accurate=closeEnough(s_fi, s_cache->fi, m->nsensordata,dell, dtheta);
    }
        if(y_accurate&s_accurate) break;
    }


}

static void mjd_stepWASPDq(const mjModel* m,
                           const mjtNum* fullstate,
                           const mjtNum* next,
                           const mjtNum* sensor,
                           mjData* d,
                           mjtNum eps, mjtByte flg_centered,
                           int skipsensor, unsigned int restore_spec,
                           mjtNum dtheta, mjtNum dell, int max_n,
                           mjtNum* next_plus, mjtNum* next_minus,  mjtNum* sensor_plus, mjtNum* sensor_minus,
                           mjtNum* y_res,  mjtNum* s_res,
                           mjtNum* y_fi, mjtNum* s_fi,
                           mjWASPCache* y_cache, mjWASPCache* s_cache){
    for(int i=0; i<max_n ;i++){
    size_t iq = y_cache->i;
    mjtNum *dpos  = mj_stackAllocNum(d, m->nv);  // allocate position perturbation
    // nudge forward
    mju_zero(dpos, m->nv);
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
        mju_zero(dpos, m->nv);
        dpos[iq] = 1;
        mj_integratePos(m, d->qpos, dpos, -eps);

        // step, get nudged output
        mj_stepSkip(m, d, mjSTAGE_NONE, skipsensor);
        getState(m, d, next_minus, sensor_minus);

        // reset
        mj_setState(m, d, fullstate, restore_spec);
    }

    // difference states
        mjtByte y_accurate=1, s_accurate=1;
    if (y_cache) {
        mju_copy(y_fi, y_cache->fi, 2*m->nv+m->na);
        if (!flg_centered) {
            stateDiff(m, y_cache->fi, next, next_plus, eps);
        } else {
            stateDiff(m, y_cache->fi, next_minus, next_plus, 2*eps);
        }
        waspUpdate(d, y_res, y_cache, 2*m->nv+m->na, m->nv);
        y_accurate=closeEnough(y_fi, y_cache->fi, 2*m->nv+m->na,dell, dtheta);
    }

    // difference sensors
    if (s_cache) {
        mju_copy(s_fi, s_cache->fi, 2*m->nsensordata);
        if (!flg_centered) {
            diff(s_cache->fi, sensor, sensor_plus, eps, m->nsensordata);
        } else {
            diff(s_cache->fi, sensor_minus, sensor_plus, 2*eps, m->nsensordata);
        }
        waspUpdate(d, s_res, s_cache, m->nsensordata, m->nv);
        s_accurate=closeEnough(s_fi, s_cache->fi,  m->nsensordata, dell, dtheta);
    }
        if(y_accurate&s_accurate) break;
    }

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
//     DyDq: (nx x nv)
//     DyDv: (nv x nv)
//     DyDa: (nv x na)
//     DyDu: (nv x nu)
//     DsDq: (ns x nv)
//     DsDv: (ns x nv)
//     DsDa: (ns x na)
//     DsDu: (ns x nu)
void mjd_transitionWASP(const mjModel* m, mjData* d, mjtNum eps, mjtByte flg_centered,
                        mjtNum q_dtheta, mjtNum q_dell, int q_max_n,
                        mjtNum v_dtheta, mjtNum v_dell, int v_max_n,
                        mjtNum a_dtheta, mjtNum a_dell, int a_max_n,
                        mjtNum u_dtheta, mjtNum u_dell, int u_max_n,
                        mjtNum* A, mjtNum* B, mjtNum* C, mjtNum* D,
                        mjWASPCache* DyDq, mjWASPCache* DyDv, mjWASPCache* DyDa,
                        mjWASPCache* DyDu,
                        mjWASPCache* DsDq, mjWASPCache* DsDv, mjWASPCache* DsDa,
                        mjWASPCache* DsDu){

    if (m->opt.integrator == mjINT_RK4) {
        mjERROR("RK4 integrator is not supported");
    }

    int nq = m->nq, nv = m->nv, na = m->na, nu = m->nu, ns = m->nsensordata;
    int ndx = 2*nv+na;

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

    // caches
    mjtNum* y_fi = mj_stackAllocNum(d, ndx);
    mjtNum* s_fi = mj_stackAllocNum(d, ns);

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
    if (DyDu || DsDu) mjd_stepWASPDu(m, fullstate, next, sensor, ctrl,
                                     d,
                                     eps, flg_centered, skipsensor, restore_spec,
                                     u_dtheta, u_dell ,u_max_n,
                                     next_plus, next_minus, sensor_plus, sensor_minus, B, D,
                                     y_fi, s_fi,
                                     DyDu, DsDu);

    // wasp-difference activations: skip=mjSTAGE_VEL
    if (DyDa || DsDa) mjd_stepWASPDa(m, fullstate, next, sensor,
                                     d,
                                     eps, flg_centered, skipsensor, restore_spec,
                                     a_dtheta, a_dell ,a_max_n,
                                     next_plus, next_minus, sensor_plus, sensor_minus, A, C,
                                     y_fi, s_fi,
                                     DyDa, DsDa);

    // wasp-difference velocities: skip=mjSTAGE_POS
    if (DyDv || DsDv) mjd_stepWASPDv(m, fullstate, next, sensor,
                                     d,
                                     eps, flg_centered, skipsensor, restore_spec,
                                     v_dtheta, v_dell ,v_max_n,
                                     next_plus, next_minus, sensor_plus, sensor_minus, A, C,
                                     y_fi, s_fi,
                                     DyDv, DsDv);

    // wasp-difference positions: skip=mjSTAGE_NONE
    if (DyDq || DsDq) mjd_stepWASPDq(m, fullstate, next, sensor,
                                     d,
                                     eps, flg_centered, skipsensor, restore_spec,
                                     q_dtheta, q_dell ,q_max_n,
                                     next_plus, next_minus, sensor_plus, sensor_minus, A, C,
                                     y_fi, s_fi,
                                     DyDq, DsDq);

    mj_freeStack(d);
}

// reset wasp cache
void mj_resetWASPCache(mjWASPCache* cache, int n, int m, mjtByte identity_basis) {
    if (cache) {
        if (identity_basis) {
            mju_zero(cache -> Delta_X, n*n);
            for (int i=0; i < n; i++) {cache->Delta_X[i*n+i] = 1.0;}
        }
        else {
            // make distribution using seed
            // Simple replacement for your C++ code
            srand(20250810);
            mjtNum* M = (mjtNum*)mju_malloc(sizeof(mjtNum)*n*n);
            for (int i = 0; i < n*n; i++) {
                M[i] = (double)rand() / RAND_MAX;  // Random values between 0 and 1
            }
            mjtNum* R = (mjtNum*) mju_malloc(n*n*sizeof(mjtNum));
            mju_qrDecompose(cache->Delta_X, R, M, n,n);
            mju_free(M);
            mju_free(R);
        }
        mju_zero(cache -> C1, n*m*n);
        mju_zero(cache -> C2, n*n);
        mju_zero(cache -> F_hat, m*n);
        mju_zero(cache -> fi, m);
        cache->i = 0;
    }
}

// delete wasp cache
void mj_deleteWASPCache(mjWASPCache* cache) {
    mju_free(cache->Delta_X);
    cache -> Delta_X = NULL;
    mju_free(cache->C1);
    cache -> C1 = NULL;
    mju_free(cache->C2);
    cache -> C2 = NULL;
    mju_free(cache->F_hat);
    cache -> F_hat = NULL;
    mju_free(cache->fi);
    cache -> fi = NULL;
    mju_free(cache);
}

// allocate wasp cache
mjWASPCache* mj_newWASPCache(int n, int m, mjtByte identity_basis) {
    mjWASPCache* cache  = (mjWASPCache*) mju_malloc(sizeof(mjWASPCache));
    cache -> Delta_X = (mjtNum*) mju_malloc(n*n*sizeof(mjtNum));
    cache -> C1  = (mjtNum*) mju_malloc(n*m*n*sizeof(mjtNum));
    cache -> C2 = (mjtNum*) mju_malloc(n*n*sizeof(mjtNum));
    cache -> F_hat = (mjtNum*) mju_malloc(m*n*sizeof(mjtNum));
    cache -> fi = (mjtNum*) mju_malloc(m*sizeof(mjtNum));
    mj_resetWASPCache(cache, n, m, identity_basis);
    return cache;
}
