
#include <glog/logging.h>
#include <benchmark/benchmark.h>

#include <Eigen/SparseCholesky>
#include <Eigen/Cholesky>

// void BM_Sparse_LDLT_Identity(benchmark::State& state) {
//   const int n = state.range_x();
//   Eigen::SparseMatrix<double> I(n, n);
//   Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
//   I.setIdentity();
//   I = 3*I;
//   while (state.KeepRunning()) {
//     solver.compute(I);
//     CHECK_EQ(Eigen::Success, solver.info());
//   }
// }
// BENCHMARK(BM_Sparse_LDLT_Identity)->Range(2<<10, 2<<15);

// void BM_DenseMatrixMultiply(benchmark::State& state) {
//   srand(0);
//   Eigen::MatrixXd A = Eigen::MatrixXd::Random(10000, 3000);
//   while (state.KeepRunning()) {
//     Eigen::MatrixXd ATA = A.transpose()*A;
//   }
// }
// BENCHMARK(BM_DenseMatrixMultiply);

void BM_DenseCholesky(benchmark::State& state) {
  srand(0);
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(3000, 3000);
  Eigen::MatrixXd ATA = A.transpose()*A;
  while (state.KeepRunning()) {
    Eigen::LLT<Eigen::MatrixXd> llt;
    llt.compute(ATA);
  }
}
BENCHMARK(BM_DenseCholesky);

BENCHMARK_MAIN();
