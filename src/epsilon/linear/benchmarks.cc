
#include <glog/logging.h>
#include <benchmark/benchmark.h>

#include <Eigen/SparseCholesky>

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

void BM_DenseMatrixMultiply(benchmark::State& state) {
  LOG(INFO) << "threads: " << Eigen::nbThreads();
  srand(0);
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(10000, 3000);
  while (state.KeepRunning()) {
    Eigen::MatrixXd ATA = A.transpose()*A;
  }
}
BENCHMARK(BM_DenseMatrixMultiply);

BENCHMARK_MAIN();
