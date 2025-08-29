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
static void getState(const mjModel *m, const mjData *d, mjtNum *state, mjtNum *sensordata) {
    mj_getState(m, d, state, mjSTATE_PHYSICS);
    if (sensordata) {
        mju_copy(sensordata, d->sensordata, m->nsensordata);
    }
}


// dx = (x2 - x1) / h
static void diff(mjtNum *restrict dx, const mjtNum *x1, const mjtNum *x2, mjtNum h, int n) {
    mjtNum inv_h = 1 / h;
    for (int i = 0; i < n; i++) {
        dx[i] = inv_h * (x2[i] - x1[i]);
    }
}


// finite-difference two state vectors ds = (s2 - s1) / h
static void stateDiff(const mjModel *m, mjtNum *ds, const mjtNum *s1, const mjtNum *s2, mjtNum h) {
    int nq = m->nq, nv = m->nv, na = m->na;

    if (nq == nv) {
        diff(ds, s1, s2, h, nq + nv + na);
    } else {
        mj_differentiatePos(m, ds, h, s1, s2);
        diff(ds + nv, s1 + nq, s2 + nq, h, nv + na);
    }
}


// finite-difference two vectors, forward, backward or centered
static void clampedDiff(mjtNum *dx, const mjtNum *x, const mjtNum *x_plus, const mjtNum *x_minus,
                        mjtNum h, int nx) {
    if (x_plus && !x_minus) {
        // forward differencing
        diff(dx, x, x_plus, h, nx);
    } else if (!x_plus && x_minus) {
        // backward differencing
        diff(dx, x_minus, x, h, nx);
    } else if (x_plus && x_minus) {
        // centered differencing
        diff(dx, x_plus, x_minus, 2 * h, nx);
    } else {
        // differencing failed, write zeros
        mju_zero(dx, nx);
    }
}

static mjtByte closeEnough(const mjtNum *a, const mjtNum *b, int n, mjtNum dell, mjtNum dtheta) {
    mjtNum dot = mju_dot(a, b, n);
    mjtNum a_norm = mju_norm(a, n);
    mjtNum b_norm = mju_norm(b, n);
    if (a_norm < 1e-10 || b_norm < 1e-10) return 0;
    if (mju_abs(dot / a_norm / b_norm - 1) > dtheta)
        return 0;
    if (mju_min(mju_abs(a_norm / b_norm - 1), mju_abs(b_norm / a_norm - 1)) > dell)
        return 0;
    return 1;
}

// finite-difference two state vectors, forward, backward or centered
static void clampedStateDiff(const mjModel *m, mjtNum *ds, const mjtNum *s, const mjtNum *s_plus,
                             const mjtNum *s_minus, mjtNum h) {
    if (s_plus && !s_minus) {
        // forward differencing
        // printf("forward diff\n");
        stateDiff(m, ds, s, s_plus, h);
    } else if (!s_plus && s_minus) {
        // backward differencing
        // printf("backward diff\n");
        stateDiff(m, ds, s_minus, s, h);
    } else if (s_plus && s_minus) {
        // centered differencing
        // printf("central diff\n");
        stateDiff(m, ds, s_minus, s_plus, 2 * h);
    } else {
        // differencing failed, write zeros
        mju_zero(ds, 2 * m->nv + m->na);
    }
}

// check if two numbers are inside a given range
static int inRange(const mjtNum x1, const mjtNum x2, const mjtNum *range) {
    return x1 >= range[0] && x1 <= range[1] &&
           x2 >= range[0] && x2 <= range[1];
}

// dim(res) = n x m
// D^T = C1 * F_hat^T + C2 * f_i^T
// F_hat = D * Delta_X
static void waspUpdate(mjData *d, mjtNum *res, mjWASPCache *cache, int m, int n) {
    size_t i = cache->i;
    mjtNum *tmp = mj_stackAllocNum(d, n * m);
    mju_mulMatMat(res, cache->C1 + i * n * n, cache->F_hat_T, n, n, m);
    mju_mulMatMatT(tmp, cache->C2 + i * n, cache->fi, n, 1, m);
    mju_addToMat(res, tmp, n, m);
    mju_mulMatTMat(cache->F_hat_T, cache->Delta_X, res, n, n, m);
    cache->i = (i + 1) % n;
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
    mjtNum* delta_u_fwd = mj_stackAllocNum(d, m->nu), *delta_u_back=mj_stackAllocNum(d, m->nu);

    for(int i=0; i<max_n ;i++){
            int fwd_exceeds=0, back_exceeds=0; // =1 when there's exceeding along 1 axis
            //printf("%d of %d\n",i, max_n);
            // check whether forward exceeds limit
            for (int j=0; j<m->nu;j++) {
                int limited = m->actuator_ctrllimited[j];
                delta_u_fwd[j] =  y_cache?(y_cache->C2+y_cache->i*m->nu)[j]*eps:(s_cache->C2+s_cache->i*m->nu)[j]*eps;
               // printf("delta_u=%f\n",delta);
                mjtNum u0 = mju_min(ctrl[j], ctrl[j]+delta_u_fwd[j]), u1=mju_max(ctrl[j], ctrl[j]+delta_u_fwd[j]);
                if (limited&& !inRange(u0, u1, m->actuator_ctrlrange+2*j)) {
                    delta_u_fwd[j]=0.0;
                    fwd_exceeds=1;
                   // printf("%d fwd exceeds\n",j);
                }
            }
            //if (y_cache) mju_printMat(y_cache->C2+y_cache->i*m->nu, 1,m->nu);
        //    mju_printMat(delta_u_fwd, 1, m->nu);
           // check whether backward exceeds limit
            for (int j=0; j<m->nu;j++) {
            int limited = m->actuator_ctrllimited[j];
            delta_u_back[j] =  y_cache?(y_cache->C2+y_cache->i*m->nu)[j]*(-eps):(s_cache->C2+s_cache->i*m->nu)[j]*(-eps);
            mjtNum u0 = mju_min(ctrl[j], ctrl[j]+delta_u_back[j]), u1=mju_max(ctrl[j], ctrl[j]+delta_u_back[j]);
            if (limited&& !inRange(u0, u1, m->actuator_ctrlrange+2*j)) {
                delta_u_back[j]=0.0;
                back_exceeds=1;
                //printf("%d backward exceeds\n",j);
            }
        }
        int nudge_fwd = !mju_isZero(delta_u_fwd, m->nu) // don't nudge forward when delta_u is all zero
        &&!(fwd_exceeds&&!back_exceeds); // fall to backward differencing when fwd exceeds but backward is accurate
            if (nudge_fwd) {
               // printf("nudging forward!\n");
                mju_addTo(d->ctrl, delta_u_fwd, m->nu);
                // step, get nudged output
                mj_stepSkip(m, d, mjSTAGE_VEL, skipsensor);
                getState(m, d, next_plus, sensor_plus);
                // reset
                mj_setState(m, d, fullstate, restore_spec);
            }
        int nudge_back = !mju_isZero(delta_u_back, m->nu) // don't nudge backward when delta_u is all zero
        &&!(!flg_centered&&!fwd_exceeds) // no need to nudge backward when using forward differencing and forward is accurate
        // (forward differencing falls to central differencing whenever fwd exceeds)
        &&!(!fwd_exceeds&&back_exceeds); // fall to forward differencing when backward exceeds but forward is accurate
            if (nudge_back) {
                //printf("nudging back!\n");
                mju_addTo(d->ctrl, delta_u_back, m->nu);
                // step, get nudged output
                mj_stepSkip(m, d, mjSTAGE_VEL, skipsensor);
                getState(m, d, next_minus, sensor_minus);
                // reset
                mj_setState(m, d, fullstate, restore_spec);
            }


        // difference states
        mjtByte y_accurate=1, s_accurate=1;
        if (y_cache) {
            mju_copy(y_fi, y_cache->F_hat_T+(y_cache->i)*(2*m->nv+m->na), 2*m->nv+m->na);
            clampedStateDiff(m, y_cache->fi, next, nudge_fwd ? next_plus : NULL,
                           nudge_back ? next_minus : NULL, eps);
            waspUpdate(d, y_res, y_cache, 2*m->nv+m->na, m->nu);
            y_accurate=closeEnough(y_fi, y_cache->fi, 2*m->nv+m->na, dell, dtheta);
        }

        // difference sensors
        if (s_cache) {
            mju_copy(s_fi, s_cache->F_hat_T+(s_cache->i)*(m->nsensordata), m->nsensordata);
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
    mjtNum* delta_v = mj_stackAllocNum(d, m->nv);
    for(int i=0; i<max_n ;i++){
    //size_t iv = y_cache?y_cache->i:s_cache->i;
    // nudge forward
    //d->qvel[iv] += eps;
    if (y_cache) {
            mju_scl(delta_v, y_cache->C2+y_cache->i*m->nv, eps, m->nv);
        }
   else {
            mju_scl(delta_v, s_cache->C2+s_cache->i*m->nv, eps, m->nv);
        }
        mju_addTo(d->qvel, delta_v, m->nv);

    // step, get nudged output
    mj_stepSkip(m, d, mjSTAGE_POS, skipsensor);
    getState(m, d, next_plus, sensor_plus);
    // reset
    mj_setState(m, d, fullstate, restore_spec);

    // nudge backward
    if (flg_centered) {
        // nudge
        //d->qvel[iv] -= eps;
        if (y_cache) {
            mju_scl(delta_v, y_cache->C2+y_cache->i*m->nv, -eps, m->nv);
        }
        else {
            mju_scl(delta_v, s_cache->C2+s_cache->i*m->nv, -eps, m->nv);
        }
        mju_addTo(d->qvel, delta_v, m->nv);
        // step, get nudged output
        mj_stepSkip(m, d, mjSTAGE_POS, skipsensor);
        getState(m, d, next_minus, sensor_minus);

        // reset
        mj_setState(m, d, fullstate, restore_spec);
    }

    // difference states
    mjtByte y_accurate=1, s_accurate=1;
    if (y_cache) {
        mju_copy(y_fi, y_cache->F_hat_T+(y_cache->i)*(2*m->nv+m->na), 2*m->nv+m->na);
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
        mju_copy(s_fi, s_cache->F_hat_T+(s_cache->i)*(m->nsensordata), m->nsensordata);
        if (!flg_centered) {
            diff(s_cache->fi, sensor, sensor_plus, eps, m->nsensordata);
        } else {
            diff(s_cache->fi, sensor_minus, sensor_plus, 2*eps, m->nsensordata);
        }
        waspUpdate(d, s_res, s_cache, m->nsensordata,  m->nv);
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
    mjtNum* delta_a = mj_stackAllocNum(d, m->na);
    for(int i=0; i<max_n ;i++){
    //size_t ia = y_cache?y_cache->i:s_cache->i;
    // nudge forward
    //d->act[ia] += eps;
        if (y_cache) {
            mju_scl(delta_a, y_cache->C2+y_cache->i*m->na, eps, m->na);
        }
        else {
            mju_scl(delta_a, s_cache->C2+s_cache->i*m->na, eps, m->na);
        }
        mju_addTo(d->ctrl, delta_a, m->na);

    // step, get nudged output
    mj_stepSkip(m, d, mjSTAGE_VEL, skipsensor);
    getState(m, d, next_plus, sensor_plus);

    // reset
    mj_setState(m, d, fullstate, restore_spec);

    // nudge backward
    if (flg_centered) {
        // nudge backward
        //d->act[ia] -= eps;
        if (y_cache) {
            mju_scl(delta_a, y_cache->C2+y_cache->i*m->na, eps, m->na);
        }
        else {
            mju_scl(delta_a, s_cache->C2+s_cache->i*m->na, eps, m->na);
        }
        mju_addTo(d->ctrl, delta_a, m->na);

        // step, get nudged output
        mj_stepSkip(m, d, mjSTAGE_VEL, skipsensor);
        getState(m, d, next_minus, sensor_minus);

        // reset
        mj_setState(m, d, fullstate, restore_spec);
    }

    // difference states
    mjtByte y_accurate=1, s_accurate=1;
    if (y_cache) {
        mju_copy(y_fi, y_cache->F_hat_T+(y_cache->i)*(2*m->nv+m->na), 2*m->nv+m->na);
        if (!flg_centered) {
            stateDiff(m, y_cache->fi, next, next_plus, eps);
        } else {
            stateDiff(m, y_cache->fi, next_minus, next_plus, 2*eps);
        }
        waspUpdate(d, y_res, y_cache, 2*m->nv+m->na, m->na);
        y_accurate=closeEnough(y_fi, y_cache->fi, 2*m->nv+m->na,dell, dtheta);
    }

    // difference sensors
    if (s_cache) {
        mju_copy(s_fi, s_cache->F_hat_T+(s_cache->i)*(m->nsensordata), m->nsensordata);
        if (!flg_centered) {
            diff(s_cache->fi, sensor, sensor_plus, eps, m->nsensordata);
        } else {
            diff(s_cache->fi, sensor_minus, sensor_plus, 2*eps, m->nsensordata);
        }
        waspUpdate(d, s_res, s_cache, m->nsensordata, m->na);
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
    mjtNum* delta_q = mj_stackAllocNum(d, m->nv);
    for(int i=0; i<max_n ;i++){
    //size_t iq = y_cache?y_cache->i:s_cache->i;
    if (y_cache) mju_copy(delta_q, y_cache->C2+y_cache->i*m->nv, m->nv);
    else mju_copy(delta_q, s_cache->C2+s_cache->i*m->nv, m->nv);
    mj_integratePos(m, d->qpos, delta_q, eps);

    // step, get nudged output
    mj_stepSkip(m, d, mjSTAGE_NONE, skipsensor);
    getState(m, d, next_plus, sensor_plus);

    // reset
    mj_setState(m, d, fullstate, restore_spec);

    // nudge backward
    if (flg_centered) {
        // nudge backward
        mj_integratePos(m, d->qpos, delta_q, -eps);

        // step, get nudged output
        mj_stepSkip(m, d, mjSTAGE_NONE, skipsensor);
        getState(m, d, next_minus, sensor_minus);

        // reset
        mj_setState(m, d, fullstate, restore_spec);
    }

    // difference states
        mjtByte y_accurate=1, s_accurate=1;
    if (y_cache) {
        mju_copy(y_fi, y_cache->F_hat_T+(y_cache->i)*(2*m->nv+m->na), 2*m->nv+m->na);
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
        mju_copy(s_fi, s_cache->F_hat_T+(s_cache->i)*(m->nsensordata), m->nsensordata);
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
//      ndx = nv + nv + na (position + velocity + activation)
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

    mjtNum* AT=NULL, *BT=NULL, *CT=NULL, *DT=NULL;
    if (A) AT = mj_stackAllocNum(d, ndx*ndx);
    if (B) BT = mj_stackAllocNum(d, nu*ndx);
    if (C) CT = mj_stackAllocNum(d, ndx*ns);
    if (D) DT = mj_stackAllocNum(d, nu*ns);
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
    // the order of differentiation wrt independent variables matters for getting correct sensor derivatives
    // the order should be u->a->v->q
    // wasp-difference controls: skip=mjSTAGE_VEL, handle ctrl at range limits
    if (DyDu || DsDu) mjd_stepWASPDu(m, fullstate, next, sensor, ctrl,
                                     d,
                                     eps, flg_centered, skipsensor, restore_spec,
                                     u_dtheta, u_dell ,mju_min(u_max_n,nu),
                                     next_plus, next_minus, sensor_plus, sensor_minus, BT, DT,
                                     y_fi, s_fi,
                                     DyDu, DsDu);

    // wasp-difference activations: skip=mjSTAGE_VEL
    if (DyDa || DsDa) mjd_stepWASPDa(m, fullstate, next, sensor,
                                     d,
                                     eps, flg_centered, skipsensor, restore_spec,
                                     a_dtheta, a_dell ,mju_min(a_max_n,na),
                                     next_plus, next_minus, sensor_plus, sensor_minus, AT?(AT+2*nv*ndx):NULL, CT?(CT+2*nv*ns):NULL,
                                     y_fi, s_fi,
                                     DyDa, DsDa);

    // wasp-difference velocities: skip=mjSTAGE_POS
    if (DyDv || DsDv) mjd_stepWASPDv(m, fullstate, next, sensor,
                                     d,
                                     eps, flg_centered, skipsensor, restore_spec,
                                     v_dtheta, v_dell ,mju_min(v_max_n,nv),
                                     next_plus, next_minus, sensor_plus, sensor_minus, AT?(AT+nv*ndx):NULL, CT?(CT+nv*ns):NULL,
                                     y_fi, s_fi,
                                     DyDv, DsDv);

    // wasp-difference positions: skip=mjSTAGE_NONE
    if (DyDq || DsDq) mjd_stepWASPDq(m, fullstate, next, sensor,
                                     d,
                                     eps, flg_centered, skipsensor, restore_spec,
                                     q_dtheta, q_dell ,mju_min(q_max_n,nv),
                                     next_plus, next_minus, sensor_plus, sensor_minus, AT, CT,
                                     y_fi, s_fi,
                                     DyDq, DsDq);



    if (A) mju_transpose(A, AT, ndx, ndx);
    if (B) mju_transpose(B, BT, nu, ndx);
    if (C) mju_transpose(C, CT, ndx, ns);
    if (D) mju_transpose(D, DT, nu, ns);

    mj_freeStack(d);
}

void mjd_transitionWASPOneThread(const mjModel *m, mjData *d, mjtNum eps, mjtByte flg_centered,
                                 mjtNum dtheta, mjtNum dell, int max_n,
                                 mjtNum *deriv,
                                 mjWASPCache *cache, mjPartialDerivativeType type) {

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
    int skipsensor = (type==mjDsDq||type==mjDsDv||type==mjDsDa||type==mjDsDu)?0:1;
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

    mjtNum* derivT=NULL;
    mjtNum* fi = (type==mjDsDq||type==mjDsDv||type==mjDsDa||type==mjDsDu)?mj_stackAllocNum(d, ns): mj_stackAllocNum(d, ndx);
    switch (type) {
        case mjDyDu:
            derivT=mj_stackAllocNum(d, nu*ndx);
            mjd_stepWASPDu(m, fullstate, next, sensor, ctrl,
                                    d,
                                    eps, flg_centered, skipsensor, restore_spec,
                                    dtheta, dell ,mju_min(max_n,nu),
                                    next_plus, next_minus, sensor_plus, sensor_minus, derivT, NULL,
                                    fi, NULL,
                                    cache, NULL);
            mju_transpose(deriv, derivT, nu, ndx);
            break;
        case mjDyDv:
            derivT=mj_stackAllocNum(d, nv*ndx);
            mjd_stepWASPDv(m, fullstate, next, sensor,
                                   d,
                                   eps, flg_centered, skipsensor, restore_spec,
                                   dtheta, dell ,mju_min(max_n,nv),
                                   next_plus, next_minus, sensor_plus, sensor_minus, derivT, NULL,
                                   fi, NULL,
                                   cache, NULL);
            mju_transpose(deriv, derivT, nv, ndx);
            break;
        case mjDyDa:
            derivT=mj_stackAllocNum(d, na*ndx);
            mju_transpose(deriv, derivT, na, ndx);
            mjd_stepWASPDa(m, fullstate, next, sensor,
                                   d,
                                   eps, flg_centered, skipsensor, restore_spec,
                                   dtheta, dell ,mju_min(max_n,na),
                                   next_plus, next_minus, sensor_plus, sensor_minus, derivT, NULL,
                                   fi, NULL,
                                   cache, NULL);
            break;
        case mjDyDq:
            derivT=mj_stackAllocNum(d, nv*ndx);
            mju_transpose(deriv, derivT, nv, ndx);
            mjd_stepWASPDq(m, fullstate, next, sensor,
                                 d,
                                 eps, flg_centered, skipsensor, restore_spec,
                                 dtheta, dell ,mju_min(max_n,nv),
                                 next_plus, next_minus, sensor_plus, sensor_minus, derivT, NULL,
                                 fi, NULL,
                                 cache, NULL);
            break;
        case mjDsDu:
            derivT=mj_stackAllocNum(d, nu*ns);
            mju_transpose(deriv, derivT, nu, ns);
            mjd_stepWASPDu(m, fullstate, next, sensor, ctrl,
                                     d,
                                     eps, flg_centered, skipsensor, restore_spec,
                                     dtheta, dell ,mju_min(max_n,nu),
                                     next_plus, next_minus, sensor_plus, sensor_minus, NULL, derivT,
                                     NULL, fi,
                                     NULL, cache);
            break;
        case mjDsDv:
            derivT=mj_stackAllocNum(d, nv*ns);
            mju_transpose(deriv, derivT, nv, ns);
            mjd_stepWASPDv(m, fullstate, next, sensor,
                                     d,
                                     eps, flg_centered, skipsensor, restore_spec,
                                     dtheta, dell ,mju_min(max_n,nv),
                                     next_plus, next_minus, sensor_plus, sensor_minus, NULL, derivT,
                                     NULL, fi,
                                     NULL, cache);
            break;
        case mjDsDa:
            derivT=mj_stackAllocNum(d, na*ns);
            mju_transpose(deriv, derivT, na, ns);
            mjd_stepWASPDa(m, fullstate, next, sensor,
                                   d,
                                   eps, flg_centered, skipsensor, restore_spec,
                                   dtheta, dell ,mju_min(max_n,na),
                                   next_plus, next_minus, sensor_plus, sensor_minus, NULL, derivT,
                                     NULL, fi,
                                     NULL, cache);
            break;
        //case mjDsDq:
         default:
            derivT=mj_stackAllocNum(d, nv*ns);
            mju_transpose(deriv, derivT, nv, ns);
            mjd_stepWASPDq(m, fullstate, next, sensor,
                                 d,
                                 eps, flg_centered, skipsensor, restore_spec,
                                 dtheta, dell ,mju_min(max_n,nv),
                                 next_plus, next_minus, sensor_plus, sensor_minus, NULL, derivT,
                                     NULL, fi,
                                     NULL, cache);
            break;
    }
    mj_freeStack(d);
}


// reset wasp cache basis
void mj_resetWASPCacheBasis(mjWASPCache *cache, int n, mjtByte identity_basis) {
    if (cache) {
        if (identity_basis) {
            mju_zero(cache->Delta_X, n * n);
            for (int i = 0; i < n; i++) { cache->Delta_X[i * n + i] = 1.0; }
        } else {
            // uncomment to fix the seed for debugging
            //srand(20250810);
            mjtNum *M = (mjtNum *) mju_malloc(sizeof(mjtNum) * n * n);
            for (int i = 0; i < n * n; i++) {
                M[i] = (double) rand() / RAND_MAX; // Random values between 0 and 1
            }
            mjtNum *R = (mjtNum *) mju_malloc(n * n * sizeof(mjtNum));
            mju_qrDecompose(cache->Delta_X, R, M, n, n);
            mju_free(M);
            mju_free(R);
        }
        // C2 = Delta_X^T
        mju_transpose(cache->C2, cache->Delta_X, n, n);
        mjtNum *tmp = (mjtNum *) mju_malloc(n * n * sizeof(mjtNum));
        mjtNum *I = (mjtNum *) mju_malloc(n * n * sizeof(mjtNum));
        mju_transpose(cache->C2, cache->Delta_X, n, n);
        for (int i = 0; i < n; i++) {
            mju_mulMatMatT(tmp, cache->C2 + i * n, cache->C2 + i * n, n, 1, n);
            // set to identity
            mju_zero(I, n * n);
            for (int j = 0; j < n; j++) { I[j * n + j] = 1.0; }
            mju_subFrom(I, tmp, n * n);
            mju_mulMatMat(cache->C1 + i * n * n, I, cache->Delta_X, n, n, n);
        }
        mju_free(tmp);
        mju_free(I);
    }
}

// copy wasp cache basis
void mj_copyWASPCacheBasis(mjWASPCache *dest, const mjWASPCache *src, int n) {
    mju_copy(dest->Delta_X, src->Delta_X, n * n);
    mju_copy(dest->C1, src->C1, n * n * n);
    mju_copy(dest->C2, src->C2, n * n);
}


// delete wasp cache
void mj_deleteWASPCache(mjWASPCache *cache) {
    if (cache) {
        mju_free(cache->Delta_X);
        cache->Delta_X = NULL;
        mju_free(cache->C1);
        cache->C1 = NULL;
        mju_free(cache->C2);
        cache->C2 = NULL;
        mju_free(cache->F_hat_T);
        cache->F_hat_T = NULL;
        mju_free(cache->fi);
        cache->fi = NULL;
        mju_free(cache);
    }
}

// allocate wasp cache
mjWASPCache *mj_newWASPCache(int n, int m, mjtByte reset_basis, mjtByte identity_basis) {
    mjWASPCache *cache = (mjWASPCache *) mju_malloc(sizeof(mjWASPCache));
    cache->Delta_X = (mjtNum *) mju_malloc(n * n * sizeof(mjtNum));
    cache->C1 = (mjtNum *) mju_malloc(n * n * n * sizeof(mjtNum));
    cache->C2 = (mjtNum *) mju_malloc(n * n * sizeof(mjtNum));
    cache->F_hat_T = (mjtNum *) mju_malloc(n * m * sizeof(mjtNum));
    cache->fi = (mjtNum *) mju_malloc(m * sizeof(mjtNum));
    mj_zeroWASPCache(cache, n, m);
    if (reset_basis) mj_resetWASPCacheBasis(cache, n, identity_basis);
    return cache;
}

// zero wasp cache (zero Fhat, fi, and i)
void mj_zeroWASPCache(mjWASPCache *cache, int n, int m) {
    cache->i = 0;
    mju_zero(cache->F_hat_T, n * m);
    mju_zero(cache->fi, m);
}
