//
// Created by dylanmac on 8/20/25.
//
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "src/engine/engine_core_smooth.h"
#include "src/engine/engine_derivative.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mujoco.h>

//#include <oneapi/tbb/task_group.h>

#include "src/engine/engine_derivative_fd.h"
// #include "src/engine/engine_derivative_wasp.h"
// #include "src/engine/engine_forward.h"
// #include "src/engine/engine_io.h"
#include "src/engine/engine_derivative_wasp.h"
#include "src/engine/engine_util_blas.h"
#include "src/engine/engine_util_errmem.h"
#include "test/fixture.h"

namespace mujoco {
namespace {
using ::testing::DoubleNear;
using ::testing::Each;
using ::testing::Eq;
using ::testing::NotNull;
using ::testing::Pointwise;
using DerivativeWASPTest = MujocoTest;
static const mjtByte use_wasp_identity_basis = 1;

// Analytic transition matrices for linear dynamical system xn = A*x + B*u
//   given modified mass matrix H (`data->qH`) and
//   Ac = H^-1 [diag(-stiffness) diag(-damping)]
//   we have
//   A  = eye(2*nv) + dt [dt*Ac + [zeros(3) eye(3)]; Ac]
//   given the moment arm matrix K (`data->actuator_moment`) and Bc = H^-1 K
//   B  = dt*[Bc*dt; Bc]
static void LinearSystem(const mjModel *m, mjData *d, mjtNum *A, mjtNum *B) {
  int nv = m->nv, nu = m->nu;
  mjtNum dt = m->opt.timestep;
  mj_markStack(d);

  // === state-transition matrix A
  if (A) {
    mjtNum *Ac = mj_stackAllocNum(d, 2 * nv * nv);
    // Ac = H^-1 [diag(-stiffness) diag(-damping)]
    mju_zero(Ac, 2 * nv * nv);
    for (int i = 0; i < nv; i++) {
      Ac[i * nv + i] = -m->jnt_stiffness[i];
      Ac[nv * nv + i * nv + i] = -m->dof_damping[i];
    }
    mj_solveLD(m, Ac, 2 * nv, d->qH, d->qHDiagInv);

    // A = [dt*Ac; Ac]
    mju_transpose(A, Ac, 2 * nv, nv);
    mju_scl(A, A, dt, nv * 2 * nv);
    mju_transpose(A + 2 * nv * nv, Ac, 2 * nv, nv);

    // Add eye(nv) to top right quadrant of A
    for (int i = 0; i < nv; i++) {
      A[i * 2 * nv + nv + i] += 1;
    }

    // A *= dt
    mju_scl(A, A, dt, 2 * nv * 2 * nv);

    // A += eye(2*nv)
    for (int i = 0; i < 2 * nv; i++) {
      A[i * 2 * nv + i] += 1;
    }
  }

  // === control-transition matrix B
  if (B) {
    mjtNum *Bc = mj_stackAllocNum(d, nu * nv);
    mjtNum *BcT = mj_stackAllocNum(d, nv * nu);
    mju_copy(Bc, d->actuator_moment, nv * nu);
    mj_solveLD(m, Bc, nu, d->qH, d->qHDiagInv);
    mju_transpose(BcT, Bc, nu, nv);
    mju_scl(B, BcT, dt * dt, nu * nv);
    mju_scl(B + nu * nv, BcT, dt, nu * nv);
  }

  mj_freeStack(d);
}

static const char *const kLinearPath = "engine/testdata/derivative/linear.xml";
static const char *const kModelPath = "testdata/model.xml";

std::vector<mjtNum> AsVector(const mjtNum *array, int n) {
  return std::vector<mjtNum>(array, array + n);
}

// errors smaller than this are ignored
static const mjtNum absolute_tolerance = 1e-9;

// corrected relative error
static mjtNum RelativeError(mjtNum a, mjtNum b) {
  mjtNum nominator = mjMAX(0, mju_abs(a - b) - absolute_tolerance);
  mjtNum denominator = (mju_abs(a) + mju_abs(b) + absolute_tolerance);
  return nominator / denominator;
}

// expect two 2D arrays to have elementwise relative error smaller than eps
// return maximum absolute error
static mjtNum CompareMatrices(mjtNum *Actual, mjtNum *Expected, int nrow,
                              int ncol, mjtNum eps) {
  mjtNum max_error = 0;
  for (int i = 0; i < nrow; i++) {
    for (int j = 0; j < ncol; j++) {
      mjtNum actual = Actual[i * ncol + j];
      mjtNum expected = Expected[i * ncol + j];
      EXPECT_LT(RelativeError(actual, expected), eps)
          << "error at position (" << i << ", " << j << ")"
          << "\nexpected = " << expected << "\nactual   = " << actual
          << "\ndiff     = " << expected - actual;
      max_error = mjMAX(mju_abs(actual - expected), max_error);
    }
  }
  return max_error;
}

  // utility function for matrix printing (debug)
  // NOLINTNEXTLINE(clang-diagnostic-unused-function)
  static void PrintMatrix(const mjtNum* mat, int nrow, int ncol) {
  std::cout.precision(5);
  std::cout << "\n";
  for (int r=0; r < nrow; r++) {
    for (int c=0; c < ncol; c++) {
      std::cout << std::fixed << std::setw(9) << mat[c + r*ncol] << " ";
    }
    std::cout << "\n";
  }
}

// used for debugging cache
void printWASPCache(const mjWASPCache* cache, int n, int m, bool print_x) {
  if (print_x) {
    std::cout<<"Delta_X:"<<std::endl;
    PrintMatrix(cache->Delta_X,n,n);
    for (int i=0; i<n; i++) {
      std::cout<<"C1["<<i<<"]:"<<std::endl;
      PrintMatrix(cache->C1+i*n*n, n, n);
    }
    std::cout<<"C2:"<<std::endl;
    PrintMatrix(cache->C2,n,n);
  }
  std::cout<<"F_hat:"<<std::endl;
  PrintMatrix(cache->F_hat,m,n);
  std::cout<<"fi:"<<std::endl;
  PrintMatrix(cache->fi,m,1);
}

// WASP derivatives don't mutate the state
TEST_F(DerivativeWASPTest, NoStateMutationWASP) {
  const std::string xml_path = GetTestDataFilePath(kModelPath);
  mjModel *model = mj_loadXML(xml_path.c_str(), nullptr, nullptr, 0);
  ASSERT_THAT(model, NotNull());
  mjData *data0 = mj_makeData(model);
  mjData *data = mj_makeData(model);
  int nv = model->nv, nu = model->nu, na = model->na, ns = model->nsensordata;

  // set time
  data->time = data0->time = 0.5;

  for (int i = 0; i < nv; i++) {
    data->qpos[i] = data0->qpos[i] = (mjtNum)i + 1;
    data->qvel[i] = data0->qvel[i] = (mjtNum)i + 2;
  }

  // set ctrl
  for (int i = 0; i < nu; i++) {
    data->ctrl[i] = data0->ctrl[i] = (mjtNum)i + 1;
  }

  // set act
  for (int i = 0; i < na; i++) {
    data->act[i] = data0->act[i] = (mjtNum)i + 1;
  }

  // allocate Jacobians, call derivatives
  int ndx = nv + nv + na;
  mjtNum *A = (mjtNum *)mju_malloc(sizeof(mjtNum) * ndx * ndx);
  mjtNum *B = (mjtNum *)mju_malloc(sizeof(mjtNum) * ndx * nu);
  mjtNum *C = (mjtNum *)mju_malloc(sizeof(mjtNum) * ns * ndx);
  mjtNum *D = (mjtNum *)mju_malloc(sizeof(mjtNum) * ns * nu);
  mjWASPCache *DyDq = mj_newWASPCache(nv, ndx, use_wasp_identity_basis);
  mjWASPCache *DyDv = mj_newWASPCache(nv, ndx, use_wasp_identity_basis);
  mjWASPCache *DyDa = mj_newWASPCache(na, ndx, use_wasp_identity_basis);
  mjWASPCache *DyDu = mj_newWASPCache(nu, ndx, use_wasp_identity_basis);
  mjWASPCache *DsDq = mj_newWASPCache(nv, ns, use_wasp_identity_basis);
  mjWASPCache *DsDv = mj_newWASPCache(nv, ns, use_wasp_identity_basis);
  mjWASPCache *DsDa = mj_newWASPCache(na, ns, use_wasp_identity_basis);
  mjWASPCache *DsDu = mj_newWASPCache(nu, ns, use_wasp_identity_basis);
  mjtNum eps = 1e-6, tol = 1e-10;
  mjd_transitionWASP(model, data, eps, /*centered=0*/
                     0, tol, tol, nv, tol, tol, nv, tol, tol, na, tol, tol, nu,
                     A, B, C, D, DyDq, DyDv, DyDa, DyDu, DsDq, DsDv, DsDa,
                     DsDu);

  // compare states in data and data0
  EXPECT_EQ(data->time, data0->time);
  EXPECT_EQ(AsVector(data->qpos, model->nq), AsVector(data0->qpos, model->nq));
  EXPECT_EQ(AsVector(data->qvel, nv), AsVector(data0->qvel, nv));
  EXPECT_EQ(AsVector(data->act, na), AsVector(data0->act, na));
  EXPECT_EQ(AsVector(data->ctrl, nu), AsVector(data0->ctrl, nu));

  mju_free(D);
  mju_free(C);
  mju_free(B);
  mju_free(A);
  mj_deleteWASPCache(DyDu);
  mj_deleteWASPCache(DyDa);
  mj_deleteWASPCache(DyDv);
  mj_deleteWASPCache(DyDq);
  mj_deleteWASPCache(DsDu);
  mj_deleteWASPCache(DsDa);
  mj_deleteWASPCache(DsDv);
  mj_deleteWASPCache(DsDq);
  mj_deleteData(data);
  mj_deleteData(data0);
  mj_deleteModel(model);
}

// compare WASP derivatives to analytic derivatives of linear dynamical system
TEST_F(DerivativeWASPTest, LinearSystemWASP) {
  const std::string xml_path = GetTestDataFilePath(kLinearPath);
  mjModel *model = mj_loadXML(xml_path.c_str(), nullptr, nullptr, 0);
  mjData *data = mj_makeData(model);
  int nv = model->nv, nu = model->nu;

  // set ctrl, integrate for 20 steps
  data->ctrl[0] = .1;
  data->ctrl[1] = -.1;
  for (int i = 0; i < 20; i++) {
    mj_step(model, data);
  }

  // analytic A and B
  mjtNum *A = (mjtNum *)mju_malloc(sizeof(mjtNum) * 2 * nv * 2 * nv);
  mjtNum *B = (mjtNum *)mju_malloc(sizeof(mjtNum) * 2 * nv * nu);

  LinearSystem(model, data, A, B);

  // uncomment for debugging:
  //std::cout<<"A:"<<std::endl;
 //  PrintMatrix(A, 2*nv, 2*nv);
  //std::cout<<"B:"<<std::endl;
  // PrintMatrix(B, 2*nv, nu);

  // forward differenced A and B
  mjtNum eps = 1e-6, tol = 1e-10;
  mjtNum *A_WASP = (mjtNum *)mju_malloc(sizeof(mjtNum) * 2 * nv * 2 * nv);
  mjtNum *B_WASP = (mjtNum *)mju_malloc(sizeof(mjtNum) * 2 * nv * nu);
  mjWASPCache *DyDq = mj_newWASPCache(nv, 2 * nv, use_wasp_identity_basis);
  mjWASPCache *DyDv = mj_newWASPCache(nv, 2 * nv, use_wasp_identity_basis);
  mjWASPCache *DyDu = mj_newWASPCache(nu, 2 * nv, use_wasp_identity_basis);
  // printWASPCache(DyDq, nv, 2*nv,true);
  mjd_transitionWASP(model, data, eps, /*centered=0*/
                     0, tol, tol, nv, tol, tol, nv, 0, 0, 0, tol, tol, nu,
                     A_WASP, B_WASP, nullptr, nullptr, DyDq, DyDv, nullptr,
                     DyDu, nullptr, nullptr, nullptr, nullptr);
  // uncomment for debugging:
  //std::cout<<"DyDq:"<<std::endl;
  //printWASPCache(DyDq, nv, 2*nv, false);
 // std::cout<<"DyDv:"<<std::endl;
 // printWASPCache(DyDv, nv, 2*nv, false);
  //std::cout<<"A_wasp:"<<std::endl;
  //PrintMatrix(A_WASP, 2*nv, 2*nv);
  //std::cout<<"DyDu:"<<std::endl;
  //printWASPCache(DyDu, nu, 2*nv, false);
  //std::cout<<"B_wasp:"<<std::endl;
  //PrintMatrix(B_WASP, 2*nv, nu);
  CompareMatrices(A, A_WASP, 2 * nv, 2 * nv, tol);
  std::cout<<"Finished comparing results for A."<<std::endl;
  CompareMatrices(B, B_WASP, 2 * nv, nu, tol);
  std::cout<<"Finished comparing results for B."<<std::endl;

  // central differenced A and B
  mjtNum *A_WASPc = (mjtNum *)mju_malloc(sizeof(mjtNum) * 2 * nv * 2 * nv);
  mjtNum *B_WASPc = (mjtNum *)mju_malloc(sizeof(mjtNum) * 2 * nv * nu);
  mjWASPCache *DyDqc = mj_newWASPCache(nv, 2 * nv, use_wasp_identity_basis);
  mjWASPCache *DyDvc = mj_newWASPCache(nv, 2 * nv, use_wasp_identity_basis);
  mjWASPCache *DyDuc = mj_newWASPCache(nu, 2 * nv, use_wasp_identity_basis);

  mjd_transitionWASP(model, data, eps, /*centered=1*/
                     1, tol, tol, nv, tol, tol, nv, 0, 0, 0, tol, tol, nu,
                     A_WASPc, B_WASPc, nullptr, nullptr, DyDqc, DyDvc, nullptr,
                     DyDuc, nullptr, nullptr, nullptr, nullptr);

  // uncomment for debugging:
  /*
  std::cout<<"DyDqc:"<<std::endl;
  printWASPCache(DyDqc, nv, 2*nv, true);
  std::cout<<"DyDvc:"<<std::endl;
  printWASPCache(DyDvc, nv, 2*nv, true);
  std::cout<<"A_waspc:"<<std::endl;
  PrintMatrix(A_WASPc, 2*nv, 2*nv);
  std::cout<<"DyDuc:"<<std::endl;
  printWASPCache(DyDuc, nu, 2*nv, true);
  std::cout<<"B_wasp:"<<std::endl;
  PrintMatrix(B_WASPc, 2*nv, nu);
*/
  CompareMatrices(A_WASP, A_WASPc, 2 * nv, 2 * nv, tol);
  std::cout<<"Finished comparing results for A central."<<std::endl;
  CompareMatrices(B_WASP, B_WASPc, 2 * nv, nu, tol);
  std::cout<<"Finished comparing results for B central."<<std::endl;

  mju_free(B_WASPc);
  mju_free(A_WASPc);
  mju_free(B_WASP);
  mju_free(A_WASP);
  mju_free(B);
  mju_free(A);
  mj_deleteWASPCache(DyDuc);
  mj_deleteWASPCache(DyDvc);
  mj_deleteWASPCache(DyDqc);
  mj_deleteWASPCache(DyDu);
  mj_deleteWASPCache(DyDv);
  mj_deleteWASPCache(DyDq);
  mj_deleteData(data);
  mj_deleteModel(model);
}

// check WASP ctrl derivatives at the range limit
TEST_F(DerivativeWASPTest, ClampedCtrlDerivativesWASP) {
  const std::string xml_path = GetTestDataFilePath(kLinearPath);
  mjModel *model = mj_loadXML(xml_path.c_str(), nullptr, nullptr, 0);
  mjData *data = mj_makeData(model);
  int nv = model->nv, nu = model->nu;

  // set ctrl, integrate for 20 steps
  data->ctrl[0] = .1;
  data->ctrl[1] = -.1;
  for (int i = 0; i < 20; i++) {
    mj_step(model, data);
  }

  // analytic B
  mjtNum *B = (mjtNum *)mju_malloc(sizeof(mjtNum) * 2 * nv * nu);

  LinearSystem(model, data, nullptr, B);

  // forward differenced A and B
  mjtNum eps = 1e-6, tol = 1e-10;
  mjtNum *B_WASP = (mjtNum *)mju_malloc(sizeof(mjtNum) * 2 * nv * nu);
  mjWASPCache *DyDu = mj_newWASPCache(nu, 2 * nv, use_wasp_identity_basis);
  // set ctrl to the limits, request forward differences
  data->ctrl[0] = 1;
  data->ctrl[1] = -1;
  mjd_transitionWASP(model, data, eps, /*centered=0*/
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, tol, tol, nu, nullptr,
                     B_WASP, nullptr, nullptr, nullptr, nullptr, nullptr, DyDu,
                     nullptr, nullptr, nullptr, nullptr);

  // expect WASP and analytic derivatives to be similar to eps precision
  CompareMatrices(B, B_WASP, 2 * nv, nu, eps);

  // ctrl remains at limits, request central differences
  mj_resetWASPCache(DyDu, nu, 2 * nv, use_wasp_identity_basis);
  mjd_transitionWASP(model, data, eps, /*centered=0*/
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, tol, tol, nu, nullptr,
                     B_WASP, nullptr, nullptr, nullptr, nullptr, nullptr, DyDu,
                     nullptr, nullptr, nullptr, nullptr);

  // expect WASP and analytic derivatives to be similar to eps precision
  CompareMatrices(B, B_WASP, 2 * nv, nu, eps);

  // set ctrl beyond limits, request forward differences
  data->ctrl[0] = 2;
  data->ctrl[1] = -2;
  mj_resetWASPCache(DyDu, nu, 2 * nv, use_wasp_identity_basis);
  mjd_transitionWASP(model, data, eps, /*centered=0*/
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, tol, tol, nu, nullptr,
                     B_WASP, nullptr, nullptr, nullptr, nullptr, nullptr, DyDu,
                     nullptr, nullptr, nullptr, nullptr);

  // expect derivatives to be 0
  EXPECT_THAT(AsVector(B_WASP, 2 * nv * nu), Each(Eq(0.0)));

  // expect ctrl to remain unchanged (despite internal clamping)
  EXPECT_EQ(data->ctrl[0], 2.0);
  EXPECT_EQ(data->ctrl[1], -2.0);

  // ctrl remains beyond limits, request centered differences
  mj_resetWASPCache(DyDu, nu, 2 * nv, use_wasp_identity_basis);
  mjd_transitionWASP(model, data, eps, /*centered=0*/
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, tol, tol, nu, nullptr,
                     B_WASP, nullptr, nullptr, nullptr, nullptr, nullptr, DyDu,
                     nullptr, nullptr, nullptr, nullptr);
  // expect derivatives to be 0
  EXPECT_THAT(AsVector(B_WASP, 2 * nv * nu), Each(Eq(0.0)));

  mju_free(B_WASP);
  mju_free(B);
  mj_deleteWASPCache(DyDu);
  mj_deleteData(data);
  mj_deleteModel(model);
}

// compare WASP sensor derivatives to analytic derivatives
TEST_F(DerivativeWASPTest, SensorDerivativesWASP) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="joint" type="slide"/>
        <geom size=".1"/>
      </body>
    </worldbody>

    <actuator>
      <general name="actuator" joint="joint" gainprm="3"/>
    </actuator>

    <sensor>
      <jointpos joint="joint"/>
      <jointvel joint="joint"/>
      <actuatorfrc actuator="actuator"/>
    </sensor>
  </mujoco>
  )";

  mjModel *model = LoadModelFromString(xml);
  int nv = model->nv, nu = model->nu, ns = model->nsensordata;
  mjData *data = mj_makeData(model);

  // expected analytic C and D
  mjtNum C[6] = {1, 0,
                 0, 1,
                 0, 0};

  mjtNum D[3] = {
      0,
      0,
      3,
  };

  // finite differenced C and D
  mjtNum eps = 1e-6, tol = 1e-10;
  mjtNum *C_WASP = (mjtNum *)mju_malloc(sizeof(mjtNum) * ns * 2 * nv);
  mjtNum *D_WASP = (mjtNum *)mju_malloc(sizeof(mjtNum) * ns * nu);
  mjWASPCache *DsDq = mj_newWASPCache(nv, ns, use_wasp_identity_basis);
  mjWASPCache *DsDv = mj_newWASPCache(nv, ns, use_wasp_identity_basis);
  mjWASPCache *DsDa = nullptr;//mj_newWASPCache(model->na, ns, use_wasp_identity_basis);
  mjWASPCache *DsDu = mj_newWASPCache(nu, ns, use_wasp_identity_basis);

  mjd_transitionWASP(model, data, eps, /*centered=0*/
                     0, tol, tol, nv, tol, tol, nv, tol, tol, model->na, tol,
                     tol, nu, nullptr, nullptr, C_WASP, D_WASP, nullptr,
                     nullptr, nullptr, nullptr, DsDq, DsDv, nullptr, DsDu);
  // compare expected and actual values
  // uncomment for debugging:

  //std::cout<<"DsDq:"<<std::endl;
  //printWASPCache(DsDq, nv, ns, false);
 // std::cout<<"DsDv:"<<std::endl;
  //printWASPCache(DsDv, nv, ns, false);
  //std::cout<<"C_wasp:"<<std::endl;
  //PrintMatrix(C_WASP, ns, 2*nv);
  //std::cout<<"DsDu:"<<std::endl;
 // printWASPCache(DsDu, nu, ns, true);
  //std::cout<<"D_wasp:"<<std::endl;
 // PrintMatrix(D_WASP, ns, nu);

  CompareMatrices(C_WASP, C, ns, 2 * nv, eps);
  std::cout<<"Finished comparing results for C."<<std::endl;
  CompareMatrices(D_WASP, D, ns, nu, eps);
  std::cout<<"Finished comparing results for D."<<std::endl;

  mju_free(D_WASP);
  mju_free(C_WASP);
  mj_deleteWASPCache(DsDu);
  mj_deleteWASPCache(DsDa);
  mj_deleteWASPCache(DsDv);
  mj_deleteWASPCache(DsDq);
  mj_deleteData(data);
  mj_deleteModel(model);
}

// if WASP sensor derivatives aren't requested, don't compute sensors
TEST_F(DerivativeWASPTest, SensorSkipWASP) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="joint" type="slide"/>
        <geom size=".1"/>
      </body>
    </worldbody>

    <actuator>
      <general name="actuator" joint="joint" gainprm="3"/>
    </actuator>

    <sensor>
      <jointpos joint="joint"/>
    </sensor>
  </mujoco>
  )";

  mjModel *model = LoadModelFromString(xml);
  int nv = model->nv, nu = model->nu;
  mjData *data = mj_makeData(model);

  // set a sentinel value in the sensor
  data->sensordata[0] = 1337;

  // finite differenced B
  mjtNum eps = 1e-6, tol = 1e-10;
  mjtNum *B_WASP = (mjtNum *)mju_malloc(sizeof(mjtNum) * 2 * nv * nu);
  mjWASPCache *DyDu = mj_newWASPCache(nu, 2 * nv, use_wasp_identity_basis);

  mjd_transitionWASP(model, data, eps, /*centered=0*/
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, tol, tol, nu, nullptr,
                     B_WASP, nullptr, nullptr, nullptr, nullptr, nullptr, DyDu,
                     nullptr, nullptr, nullptr, nullptr);
  EXPECT_EQ(data->sensordata[0], 1337) << "sensors should not be recomputed";
  mju_free(B_WASP);
  mj_deleteWASPCache(DyDu);
  mj_deleteData(data);
  mj_deleteModel(model);
}

} // namespace
} // namespace mujoco